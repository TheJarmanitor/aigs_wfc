import os
import json
import time
import base64

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, Http404, HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.db.models import F
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.views import generic
from django.utils import timezone
from django.core.cache import cache

import pickle
from PIL import Image
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tensorneat import genome, common

from .models import Layout, CPPNState, PureCPPNState

from cwapp.tools.SimpleGAArrays import GenerateNewPopulation
from cwapp.tools.SGAA_test import imgFromArray

from aigs.interactive_NEAT import InteractiveNEAT
from aigs.interactive_problem import InteractiveGrid
from aigs.interactive_pipeline import InteractivePipeline
from aigs.tools import rule_split, wfc, visualize_wfc, prepared_bundles

IMAGES_PER_PAGE = 10
LAYOUT_RESOLUTION = 16
HIDDEN_LAYERS = (5, 3)
ACTIVATION_FUNCTIONS = [
    common.ACT.sigmoid,
    common.ACT.tanh,
    common.ACT.sin,
    common.ACT.relu
    ]

# WFC settings
WFC_PATH = "aigs/images/tileset_inputs/dragon_warrior/dragonwarr_island.png"
#WFC_PATH = "aigs/images/tileset_inputs/dragon_warrior/dragon_warrior_map.png"
WFC_TILE_SIZE = 16
BUNDLE_WEIGHT = 100.0
BUNDLE = prepared_bundles.bundle_dragon_warr_island
WFC_SIZE = LAYOUT_RESOLUTION
STATIC_WFC_OUTPUT_PATH = "static/assets/output"

LAYOUT_COLORS = [
    [40, 229, 34],  # land
    [24, 28, 214],  # water
    [85, 10, 10],  # mountains
    [211, 26, 26],  # city
]


# Main page layout. When user opens the webpage, this method is called.
def IndexView(request, version="A"):
    template_name = "index.html"
    if version not in ["A", "B", "C","D", "a", "b", "c", "d"]:
        version = "a"
    version = version.upper()
    n = list(map(str, range(IMAGES_PER_PAGE)))
    user_id = -1

    _prepare_ruleset()
    # check if default exists
    if version == "B" or version == "D":
        n = _init_new_nocppn_population(version=version)
    elif version == "C":
        user_id = _get_new_user_id(wfc=False)
    else:
        user_id = _get_new_user_id()

    context = {
        "images": IMAGES_PER_PAGE,
        "n": n,
        "version": version,
        "user_id": user_id,
    }
    return render(request, template_name, context)


# when clicked on Mutate button, this process is called on the server
def process_images(request):
    if request.method == "POST":
        data = json.loads(request.body)  # Parse the incoming JSON

        parents_ids = data.get("selected_images", [])
        all_ids = data.get("all_images", [])
        version = data.get("version", "A")
        if version not in ["A", "B", "C", "D"]:
            version = "A"

        if version == "B" or version == "D":
            new_ids = _nocppn_process_imgs(parents_ids, all_ids,version=version)
            user_id = -2
        elif version == "C":
            user_id = data.get("user_id", -1)
            print("user_id: ", user_id)
            new_ids, user_id = _pure_cppn_procces_imgs(user_id, parents_ids)
        else:
            user_id = data.get("user_id", -1)
            print("user_id: ", user_id)
            new_ids, user_id = _cppn_process_imgs(user_id, parents_ids)

        return JsonResponse(
            {
                "message": "Images processed successfully",
                "new_images": new_ids,
                "user_id": user_id,
            }
        )

    return JsonResponse({"error": "Invalid request"}, status=400)

def delete_images(request):
    if request.method == "POST":
        data = json.loads(request.body)  # Parse the incoming JSON

        ids = data.get("ids", [])
        version = data.get("version", "_")
        user_id = data.get("user_id", -1)
        if version not in ["A", "B", "C", "D"] or user_id == -1:
            return JsonResponse({"error": "Invalid request"}, status=400)

        _delete_ids(version, user_id, ids)

        return JsonResponse(
            {
                "message": "Images deleted successfully",
            }
        )

    return JsonResponse({"error": "Invalid request"}, status=400)


def _nocppn_process_imgs(parents_ids, all_ids, version="B"):
    # get parents from database
    parents = Layout.objects.filter(id__in=parents_ids).values_list("data", flat=True)
    parents = [json.loads(p) for p in parents]

    # get new population
    offsprings = GenerateNewPopulation(
        parents, IMAGES_PER_PAGE - len(parents), 0.1, 0.5, keep_parents=False
    )

    # add each new layout to the database and get id
    offsprings_ids = []
    for offspring in offsprings:
        layout = Layout(data=json.dumps(offspring.tolist()), pub_date=timezone.now())
        layout.save()
        offsprings_ids.append(layout.id)


        path_file = f"static/assets/generated/{version}/img_{layout.id}"
        img_path = imgFromArray(
            offspring, f"{path_file}.png"
        )
        _run_wfc(img_path,f"{version}/img_{layout.id}_wfc.png",layout.id)
        layout.img_path = img_path
        layout.save()

    new_ids = []
    for id in all_ids:
        if id in parents_ids:
            new_ids.append(id)
        else:
            new_ids.append(str(offsprings_ids.pop(0)))

    # remove old layouts
    delete_ids = [
        id for id in all_ids if id not in new_ids and int(id) > IMAGES_PER_PAGE
    ]
    Layout.objects.filter(id__in=delete_ids).delete()
    for id in delete_ids:
        os.remove(f"static/assets/generated/{version}/img_{id}.png")
        os.remove(f"static/assets/generated/{version}/img_{id}_wfc.png")

    return new_ids


def _cppn_process_imgs(user_id, parents_ids):

    if user_id == -1:
        print("ERROR")
        return None

    cppnState = CPPNState.objects.get(id=user_id)
    state = _pickle_loads(cppnState.data)

    pipeline = _get_pipeline()

    # render layouts

    selected_indices = pipeline.algorithm.select_winners(parents_ids)
    state = pipeline.evolve(state, selected_indices)

    population = pipeline.generate(state)
    pipeline.visualize_population(
        population, save_path="static/assets/generated", file_name=f"img_X_{user_id}"
    )
    for i in range(IMAGES_PER_PAGE):
        if i in parents_ids:
            continue
        _run_wfc(
            f"static/assets/generated/img_X_{user_id}_{i}.png",
            f"img_X_{user_id}_{i}_wfc.png",
            i,
        )

    cppnState.data = _pickle_dumps(state)
    cppnState.pub_date = timezone.now()
    cppnState.save()

    return list(range(IMAGES_PER_PAGE)), user_id


def _pure_cppn_procces_imgs(user_id, parents_ids):
    if user_id == -1:
        print("ERROR")
        return None

    cppnState = PureCPPNState.objects.get(id=user_id)
    state = _pickle_loads(cppnState.data)

    cppn_pipeline = _get_pipeline(wfc=True)

    # render layouts

    selected_indices = cppn_pipeline.algorithm.select_winners(parents_ids)
    state = cppn_pipeline.evolve(state, selected_indices)

    population = cppn_pipeline.generate(state)
    cppn_pipeline.visualize_population(
        population,
        save_path="static/assets/output",
        file_name=f"img_Z_{user_id}",
        save_as_text=True,
    )
    for i in range(IMAGES_PER_PAGE):
        if i in parents_ids:
            continue
        _run_wfc(output_path=f"img_Z_{user_id}_{i}_wfc.png", seed=i)

    cppnState.data = _pickle_dumps(state)
    cppnState.pub_date = timezone.now()
    cppnState.save()


    return list(range(IMAGES_PER_PAGE)), user_id


def _get_pipeline(wfc=False):
    global pipeline
    global cppn_pipeline

    # check if pipeline exists
    if not wfc:
        if not "pipeline" in globals():
            print("setupping")
            test_genome = genome.DefaultGenome(
                num_inputs=2,
                num_outputs=4,
                node_gene=genome.DefaultNode(
                    activation_options=[
                        common.ACT.sigmoid,
                        common.ACT.tanh,
                        common.ACT.sin,
                    ]
                ),
                init_hidden_layers=HIDDEN_LAYERS,
                max_conns=128,
            )

            algo = InteractiveNEAT(
                pop_size=IMAGES_PER_PAGE,
                genome=test_genome,
            )

            problem = InteractiveGrid(grid_size=(LAYOUT_RESOLUTION, LAYOUT_RESOLUTION))
            grid = plt.imread("aigs/images/cppn_inputs/piskel_example1.png")

            pipeline = InteractivePipeline(
                algorithm=algo, problem=problem, input_grid=grid
            )
        return pipeline

    else:
        if not "cppn_pipeline" in globals():
            test_genome = genome.DefaultGenome(
                num_inputs=2,
                num_outputs=_prepare_ruleset(),
                node_gene=genome.DefaultNode(
                    activation_options=[
                        common.ACT.sigmoid,
                        common.ACT.tanh,
                        common.ACT.sin,
                    ]
                ),
                init_hidden_layers=HIDDEN_LAYERS,
                max_conns=256,
            )

            algo = InteractiveNEAT(
                pop_size=IMAGES_PER_PAGE,
                genome=test_genome,
            )

            problem = InteractiveGrid(grid_size=(LAYOUT_RESOLUTION, LAYOUT_RESOLUTION))
            grid = plt.imread(
                "aigs/images/tileset_inputs/dragon_warrior/dragonwarr_island.png"
            )
            cppn_pipeline = InteractivePipeline(
                algorithm=algo, problem=problem, input_grid=grid, tile_size=16
            )
        return cppn_pipeline


def _get_default_state():
    cppnstate0 = CPPNState.objects.filter(id=0)
    if (
        len(cppnstate0) > 0
        and os.path.exists("static/assets/generated/img_X_0.png")
        and os.path.exists("static/assets/generated/img_X_0_wfc.png")
    ):
        return _pickle_loads(cppnstate0[0].data)

    pipeline = _get_pipeline()
    state = pipeline.setup()
    population = pipeline.generate(state)
    pipeline.visualize_population(
        population, save_path="static/assets/generated", file_name=f"img_X"
    )
    for i in range(IMAGES_PER_PAGE):
        _run_wfc(f"static/assets/generated/img_X_{i}.png", f"img_X_{i}_wfc.png", i)
    cppnstate0 = CPPNState(data=_pickle_dumps(state), pub_date=timezone.now())
    cppnstate0.save()

    # override the 0th id
    try:
        cppnstate0 = CPPNState.objects.get(id=0)
    except:
        cppnstate0 = CPPNState(pub_date=timezone.now())
        cppnstate0.id = 0
    cppnstate0.data = _pickle_dumps(state)
    cppnstate0.pub_date = timezone.now()
    cppnstate0.save()

    return state

def _get_new_state():
    cppnstate = CPPNState(pub_date=timezone.now())
    cppnstate.save()
    user_id = cppnstate.id
    pipeline = _get_pipeline()
    state = pipeline.setup()
    pipeline.seed = user_id
    population = pipeline.generate(state)
    pipeline.visualize_population(
        population, save_path="static/assets/generated", file_name=f"img_X_{user_id}"
    )
    for i in range(IMAGES_PER_PAGE):
        _run_wfc(f"static/assets/generated/img_X_{user_id}_{i}.png", f"img_X_{user_id}_{i}_wfc.png", i)
    cppnstate.data = _pickle_dumps(state)
    cppnstate.save()

    return user_id 


def _get_cppn_state():
    cppnstate0 = PureCPPNState.objects.filter(id=0)
    if (
        len(cppnstate0) > 0
        and os.path.exists("static/assets/generated/img_Z_0.png")
        and os.path.exists("static/assets/generated/img_Z_0_wfc.png")
    ):
        return _pickle_loads(cppnstate0[0].data)

    cppn_pipeline = _get_pipeline(wfc=True)
    state = cppn_pipeline.setup()
    population = cppn_pipeline.generate(state)
    cppn_pipeline.visualize_population(
        population,
        save_path="static/assets/output",
        file_name=f"img_Z",
        save_as_text=True,
    )
    for i in range(IMAGES_PER_PAGE):
        _run_wfc(output_path=f"img_Z_{i}_wfc.png", seed=i)
    cppnstate0 = PureCPPNState(data=_pickle_dumps(state), pub_date=timezone.now())
    cppnstate0.save()

    # override the 0th id
    try:
        cppnstate0 = PureCPPNState.objects.get(id=0)
    except:
        cppnstate0 = PureCPPNState(pub_date=timezone.now())
        cppnstate0.id = 0
    cppnstate0.data = _pickle_dumps(state)
    cppnstate0.pub_date = timezone.now()
    cppnstate0.save()

    return state

def _get_new_cppn_state():
    cppnstate = PureCPPNState(pub_date=timezone.now())
    cppnstate.save()
    user_id = cppnstate.id
    cppn_pipeline = _get_pipeline(wfc=True)
    cppn_pipeline.seed = user_id
    state = cppn_pipeline.setup()
    population = cppn_pipeline.generate(state)
    cppn_pipeline.visualize_population(
        population,
        save_path="static/assets/output",
        file_name=f"img_Z_{user_id}",
        save_as_text=True,
    )
    for i in range(IMAGES_PER_PAGE):
        _run_wfc(output_path=f"img_Z_{user_id}_{i}_wfc.png", seed=i)
    cppnstate.data = _pickle_dumps(state)
    cppnstate.save()
    return user_id


def _pickle_loads(data):
    return pickle.loads(base64.b64decode(data))


def _pickle_dumps(data):
    return base64.b64encode(pickle.dumps(data)).decode("utf-8")


def _get_new_user_id(wfc=True, deterministic=False):
    if deterministic:
        if wfc:
            default_state = _get_default_state()
            cppnState = CPPNState(
                data=_pickle_dumps(default_state), pub_date=timezone.now()
            )
        else:
            default_state = _get_cppn_state()
            cppnState = PureCPPNState(
                data=_pickle_dumps(default_state), pub_date=timezone.now()
            )
        cppnState.save()
        return cppnState.id
    else:
        if wfc:
            return _get_new_state()
        else:
            return _get_new_cppn_state()


def _init_nocppn_population(version="B"):
    # check if default exists
    ok = True

    n = range(IMAGES_PER_PAGE) if version == "D" else range(IMAGES_PER_PAGE, 2*IMAGES_PER_PAGE)

    if len(Layout.objects.filter(id__in=range(IMAGES_PER_PAGE))) != IMAGES_PER_PAGE:
        ok = False
    if not os.path.exists(f"static/assets/generated/{version}/"):
        ok = False
    for i in n:
        if not os.path.exists(f"static/assets/generated/{version}/img_{i}.png") or not os.path.exists(f"static/assets/generated/{version}/img_{i}_wfc.png"):
            ok = False
            break
    n = list(map(str, n))
    if ok:
        return n

    pipeline = _get_pipeline()
    state = _get_default_state()
    if version == "D":
        pop = pipeline.generate(state)
    else:
        pop = _generate_noise((IMAGES_PER_PAGE, LAYOUT_RESOLUTION, LAYOUT_RESOLUTION, 4))
    os.makedirs(f"static/assets/generated/{version}/", exist_ok=True)
    os.makedirs(f"static/assets/output/{version}/", exist_ok=True)
    pop = jnp.reshape(pop, (IMAGES_PER_PAGE, LAYOUT_RESOLUTION, LAYOUT_RESOLUTION, 4))
    offset = 0 if version == "D" else IMAGES_PER_PAGE
    for i in range(IMAGES_PER_PAGE):
        layout = Layout(data=json.dumps(pop[i].tolist()), pub_date=timezone.now())
        layout.id = i + offset
        layout.save()
        img_path = imgFromArray(
            pop[i], f"static/assets/generated/{version}/img_{layout.id}.png"
        )
        layout.img_path = img_path
        layout.save()
        _run_wfc(img_path,f"{version}/img_{layout.id}_wfc.png",i)
    return n

def _init_new_nocppn_population(version="B"):
    pipeline = _get_pipeline()
    state = _get_default_state()
    if version == "D":
        pop = pipeline.generate(state)
    else:
        pop = _generate_noise((IMAGES_PER_PAGE, LAYOUT_RESOLUTION, LAYOUT_RESOLUTION, 4))
    os.makedirs(f"static/assets/generated/{version}/", exist_ok=True)
    os.makedirs(f"static/assets/output/{version}/", exist_ok=True)
    pop = jnp.reshape(pop, (IMAGES_PER_PAGE, LAYOUT_RESOLUTION, LAYOUT_RESOLUTION, 4))
    offset = 0 if version == "D" else IMAGES_PER_PAGE
    ids = []
    for i in range(IMAGES_PER_PAGE):
        layout = Layout(data=json.dumps(pop[i].tolist()), pub_date=timezone.now())
        layout.save()
        img_path = imgFromArray(
            pop[i], f"static/assets/generated/{version}/img_{layout.id}.png"
        )
        layout.img_path = img_path
        layout.save()
        _run_wfc(img_path,f"{version}/img_{layout.id}_wfc.png",i)
        ids.append(layout.id)
    return list(map(str, ids))

def _prepare_ruleset():
    output_folder = WFC_PATH.replace("/", "_")[:-4]
    if os.path.exists(f"{STATIC_WFC_OUTPUT_PATH}/{output_folder}") and os.path.exists(
        f"{STATIC_WFC_OUTPUT_PATH}/{output_folder}/rules.pkl"
    ):
        rules = pickle.load(open(f"{STATIC_WFC_OUTPUT_PATH}/{output_folder}/rules.pkl", "rb"))
        return len(rules)
    os.makedirs(f"{STATIC_WFC_OUTPUT_PATH}/{output_folder}", exist_ok=True)


    img = Image.open(WFC_PATH)
    img = img.convert("RGB")
    img = np.array(img)
    rules = rule_split.RuleSet(
        [list(map(lambda x: rule_split.Color(x[0], x[1], x[2]), row)) for row in img],
        WFC_TILE_SIZE,
    )
    rules.output_to_folder_rules(output_folder, output_dir=STATIC_WFC_OUTPUT_PATH)
    print(f"Created {output_folder} rules")
    return len(rules.tiles)


def _run_wfc(layout_input_path=None, output_path=None, seed=42):
    # %% Execute wfc
    rules_file = f"{STATIC_WFC_OUTPUT_PATH}/{WFC_PATH.replace('/','_')[:-4]}/rules.pkl"
    txt_wfc_file = f"{STATIC_WFC_OUTPUT_PATH}/{output_path}.txt"
    image_output = f"static/assets/generated/{output_path}"

    if layout_input_path is not None:
        rules = pickle.load(open(rules_file, "rb"))
        local_weights = wfc.local_weight(
            BUNDLE,
            default_weight=1.0,
            prob_magnitude=BUNDLE_WEIGHT,
            tile_count=len(rules),
        )

        layout = _layout_to_array(layout_input_path, LAYOUT_COLORS)

        wfc.wfc(
            [*range(len(rules))],
            rules,
            WFC_SIZE,
            WFC_SIZE,
            weights=local_weights,
            path_to_output=txt_wfc_file,
            layout_map=layout,
            seed=seed,
        )

    wfc_rule_path = f"{STATIC_WFC_OUTPUT_PATH}/{WFC_PATH.replace('/','_')[:-4]}/"
    visualize_wfc.visualize_wfc(
        path_folder=wfc_rule_path,
        input_file=txt_wfc_file,
        output_file=image_output,
        SHOW_NUKES=False,
    )

    try:
        os.remove(txt_wfc_file)
    except:
        pass


def _layout_to_array(layout_path, color_map):
    img = Image.open(layout_path)
    img = img.convert("RGB")

    color_map = [tuple(c) for c in color_map]

    out = [[0 for _ in range(LAYOUT_RESOLUTION)] for _ in range(LAYOUT_RESOLUTION)]
    for i in range(LAYOUT_RESOLUTION):
        for j in range(LAYOUT_RESOLUTION):
            color = img.getpixel((i, j))
            out[j][i] = color_map.index(color)
    return jnp.array(out)

def _generate_noise(shape):
    pop = np.random.rand(*shape)
    return pop

def _delete_ids(version, user_id, ids):
    if version=="A":
        for id in ids:
            if os.path.exists(f"static/assets/generated/img_X_{user_id}_{id}.png"):
                os.remove(f"static/assets/generated/img_X_{user_id}_{id}.png")
            if os.path.exists(f"static/assets/generated/img_X_{user_id}_{id}_wfc.png"):
                os.remove(f"static/assets/generated/img_X_{user_id}_{id}_wfc.png")
        CPPNState.objects.filter(id__in=[user_id]).delete()
    if version=="C":
        for id in ids:
            if os.path.exists(f"static/assets/generated/img_Z_{user_id}_{id}.png"):
                os.remove(f"static/assets/generated/img_Z_{user_id}_{id}.png")
            if os.path.exists(f"static/assets/generated/img_Z_{user_id}_{id}_wfc.png"):
                os.remove(f"static/assets/generated/img_Z_{user_id}_{id}_wfc.png")
        PureCPPNState.objects.filter(id__in=[user_id]).delete()
    if version=="B":
        for id in ids:
            if os.path.exists(f"static/assets/generated/{version}/img_{id}.png"):
                os.remove(f"static/assets/generated/{version}/img_{id}.png")
            if os.path.exists(f"static/assets/generated/{version}/img_{id}_wfc.png"):
                os.remove(f"static/assets/generated/{version}/img_{id}_wfc.png")
        Layout.objects.filter(id__in=ids).delete()
