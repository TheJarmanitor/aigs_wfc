from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, Http404, HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.db.models import F
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.views import generic
from django.utils import timezone

import json
import random
import time
import os

from .models import Layout, CPPNState

from cwapp.tools.SimpleGAArrays import GenerateNewPopulation
from cwapp.tools.SGAA_test import differentTest, imgFromArray
from django.core.cache import cache

import jax.numpy as jnp

from aigs.interactive_NEAT import InteractiveNEAT
from aigs.interactive_problem import InteractiveGrid
from aigs.interactive_pipeline import InteractivePipeline
from tensorneat import genome, common
import matplotlib.pyplot as plt

import base64
import pickle

<<<<<<< HEAD
IMAGES_PER_PAGE = 20
LAYOUT_RESOLUTION = 64
HIDDEN_LAYERS = (4, 4, 4)

=======
import numpy as np
from aigs.tools import rule_split, wfc, visualize_wfc, prepared_bundles
from PIL import Image

IMAGES_PER_PAGE = 10
LAYOUT_RESOLUTION = 16
HIDDEN_LAYERS = (3,4)
>>>>>>> reload

# WFC settings
WFC_PATH = "aigs/images/tileset_inputs/dragon_warrior/dragonwarr_island.png"
WFC_TILE_SIZE = 16
BUNDLE_WEIGHT = 100.0
BUNDLE = prepared_bundles.bundle_dragon_warr
WFC_SIZE = LAYOUT_RESOLUTION
STATIC_WFC_OUTPUT_PATH = "static/assets/output"

LAYOUT_COLORS = [
    [ 40, 229,  34], #land
    [ 24,  28, 214], #water
    [ 85,  10,  10], #mountains
    [ 211, 26,  26]  #city
]


# Main page layout. When user opens the webpage, this method is called.
def IndexView(request, version="A"):
    template_name = "index.html"
    if version not in ["A", "B", "C", "a", "b", "c"]:
        version = "a"
    version = version.upper()
    n = list(map(str, range(IMAGES_PER_PAGE)))
    user_id = -1

    _prepare_ruleset()
    # check if default exists
    if version == "B":
        _init_nocppn_population()
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
        if version not in ["A", "B", "C"]:
            version = "A"

        if version == "B":
            new_ids = _nocppn_process_imgs(parents_ids, all_ids)
            user_id = -2
        elif version == "C":
            user_id = data.get("user_id", -1)
            print("user_id: ", user_id)
            new_ids, user_id = _cppn_process_imgs(user_id, parents_ids, wfc=False)
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


def _nocppn_process_imgs(parents_ids, all_ids):
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

        path_file = f"static/assets/generated/img_{layout.id}"
        img_path = imgFromArray(
            offspring, f"{path_file}.png"
        )
        _run_wfc(img_path,f"{path_file}_wfc.png",layout.id)
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
        os.remove(f"static/assets/generated/img_{id}.png")
        os.remove(f"static/assets/generated/img_{id}_wfc.png")

    return new_ids


def _cppn_process_imgs(user_id, parents_ids, wfc=True):

    if user_id == -1:
        print("ERROR")
        return None
    stopwatch = time.time()

    cppnState = CPPNState.objects.get(id=user_id)
    state = _pickle_loads(cppnState.data)

    pipeline = _get_pipeline(wfc)

    # render layouts

    print("Time elapsed 1: ", time.time() - stopwatch)

    selected_indices = pipeline.algorithm.select_winners(parents_ids)
    state = pipeline.evolve(state, selected_indices)

    population = pipeline.generate(state)
    pipeline.visualize_population(
        population, save_path="static/assets/generated", file_name=f"img_{user_id}"
    )
    for i in range(IMAGES_PER_PAGE):
        if i in parents_ids:
            continue
        _run_wfc(f"static/assets/generated/img_{user_id}_{i}.png",f"img_{user_id}_{i}_wfc.png",i)

    cppnState.data = _pickle_dumps(state)
    cppnState.pub_date = timezone.now()
    cppnState.save()

    print("Time elapsed 2: ", time.time() - stopwatch)

    return list(range(IMAGES_PER_PAGE)), user_id


def _get_pipeline(wfc=True):
    global pipeline

    # check if pipeline exists
    if wfc:
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

    else:
        if not "pipeline" in globals():
            test_genome = genome.DefaultGenome(
                num_inputs=2,
                num_outputs=18,
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
            grid = plt.imread(
                "aigs/images/tileset_inputs/dragon_warrior/dragonwarr_island.png"
            )
            pipeline = InteractivePipeline(
                algorithm=algo, problem=problem, input_grid=grid, tile_size=16
            )
    return pipeline


def _get_default_state(wfc=True):
    cppnstate0 = CPPNState.objects.filter(id=0)
    if len(cppnstate0) > 0 and os.path.exists("static/assets/generated/img_X_0.png") and \
        os.path.exists("static/assets/generated/img_X_0_wfc.png"):
            return _pickle_loads(cppnstate0[0].data)



    pipeline = _get_pipeline(wfc)
    state = pipeline.setup()
    population = pipeline.generate(state)
    pipeline.visualize_population(
        population, save_path="static/assets/generated", file_name=f"img_X"
    )
    for i in range(IMAGES_PER_PAGE):
        _run_wfc(f"static/assets/generated/img_X_{i}.png",f"img_X_{i}_wfc.png",i)
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


def _pickle_loads(data):
    return pickle.loads(base64.b64decode(data))


def _pickle_dumps(data):
    return base64.b64encode(pickle.dumps(data)).decode("utf-8")


def _get_new_user_id(wfc=True):
    default_state = _get_default_state(wfc)
    cppnState = CPPNState(data=_pickle_dumps(default_state), pub_date=timezone.now())
    cppnState.save()
    return cppnState.id


def _init_nocppn_population():
    # check if default exists
    ok = True

    if len(Layout.objects.filter(id__in=range(IMAGES_PER_PAGE))) != IMAGES_PER_PAGE:
        ok = False
    for i in range(IMAGES_PER_PAGE):
        if not os.path.exists(f"static/assets/generated/img_{i}.png"):
            ok = False
            break
    if ok:
        return

    pipeline = _get_pipeline()
    state = _get_default_state()
    pop = pipeline.generate(state)
    print(pop.shape)
    pop = jnp.reshape(pop, (IMAGES_PER_PAGE, LAYOUT_RESOLUTION, LAYOUT_RESOLUTION, 4))
    for i in range(IMAGES_PER_PAGE):
        layout = Layout(data=json.dumps(pop[i].tolist()), pub_date=timezone.now())
        layout.id = i
        layout.save()
        img_path = imgFromArray(pop[i], f"static/assets/generated/img_{layout.id}.png")
        layout.img_path = img_path
        layout.save()

def _prepare_ruleset():
    output_folder = WFC_PATH.replace("/","_")[:-4]
    if os.path.exists(f"{STATIC_WFC_OUTPUT_PATH}/{output_folder}") and os.path.exists(f"{STATIC_WFC_OUTPUT_PATH}/{output_folder}/rules.pkl"):
        return
    os.makedirs(f"{STATIC_WFC_OUTPUT_PATH}/{output_folder}", exist_ok=True)

    print("Preparing ruleset")

    img = Image.open(WFC_PATH)
    img = img.convert("RGB")
    img = np.array(img)
    rules = rule_split.RuleSet([list(map(lambda x: rule_split.Color(x[0], x[1], x[2]), row)) for row in img], WFC_TILE_SIZE)
    rules.output_to_folder_rules(output_folder, output_dir = STATIC_WFC_OUTPUT_PATH)
    print(f"Created {output_folder} rules")
    return

def _run_wfc(layout_input_path,output_path,seed):
    #%% Execute wfc
    rules_file = f"{STATIC_WFC_OUTPUT_PATH}/{WFC_PATH.replace('/','_')[:-4]}/rules.pkl"
    txt_wfc_file = f"{STATIC_WFC_OUTPUT_PATH}/{output_path}.txt"
    image_output = f"static/assets/generated/{output_path}"

    rules = pickle.load(open(rules_file, "rb"))
    local_weights = wfc.local_weight(BUNDLE, default_weight=1.0,prob_magnitude=BUNDLE_WEIGHT, tile_count=len(rules))

    layout = _layout_to_array(layout_input_path, LAYOUT_COLORS)

    wfc.wfc([*range(len(rules))], rules, WFC_SIZE, WFC_SIZE,weights=local_weights, path_to_output=txt_wfc_file, layout_map = layout, seed=seed)
    wfc_rule_path = f"{STATIC_WFC_OUTPUT_PATH}/{WFC_PATH.replace('/','_')[:-4]}/"
    visualize_wfc.visualize_wfc(path_folder = wfc_rule_path, input_file = txt_wfc_file, output_file = image_output, SHOW_NUKES = False)

def _layout_to_array(layout_path, color_map):
    img = Image.open(layout_path)
    img = img.convert("RGB")

    color_map = [tuple(c) for c in color_map]
    print(color_map)
    print("---------")

    out = [[0 for _ in range(LAYOUT_RESOLUTION)] for _ in range(LAYOUT_RESOLUTION)]
    for i in range(LAYOUT_RESOLUTION):
        for j in range(LAYOUT_RESOLUTION):
            color = img.getpixel((i, j))
            out[i][j] = color_map.index(color)
    return jnp.array(out)
