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

IMAGES_PER_PAGE = 16
LAYOUT_RESOLUTION = 32

# Main page layout. When user opens the webpage, this method is called.
def IndexView(request, version="A"):
    template_name = "index.html"
    if version not in ["A", "B"]:
        version = "A"
    n = list(map(str, range(IMAGES_PER_PAGE)))
    user_id = -1

    # check if default exists
    if version == "B":
        _init_nocppn_population()
    else:
        user_id = _get_new_user_id()

    context = {"images": IMAGES_PER_PAGE, "n": n, "version": version, "user_id": user_id}
    return render(request, template_name, context)

# when clicked on Mutate button, this process is called on the server
def process_images(request):
    if request.method == "POST":
        data = json.loads(request.body)  # Parse the incoming JSON

        parents_ids = data.get("selected_images", [])
        all_ids = data.get("all_images", [])
        version = data.get("version", "A")
        if version not in ["A", "B"]:
            version = "A"

        if version == "B":
            new_ids = _nocppn_process_imgs(parents_ids, all_ids)
            user_id = -2
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
        img_path = imgFromArray(
            offspring, f"static/assets/generated/img_{layout.id}.png"
        )
        layout.img_path = img_path
        layout.save()

    new_ids = []
    for id in all_ids:
        if id in parents_ids:
            new_ids.append(id)
        else:
            new_ids.append(str(offsprings_ids.pop(0)))

    # remove old layouts
    delete_ids = [id for id in all_ids if id not in new_ids and int(id) > IMAGES_PER_PAGE]
    Layout.objects.filter(id__in=delete_ids).delete()
    for id in delete_ids:
        os.remove(f"static/assets/generated/img_{id}.png")

    return new_ids


def _cppn_process_imgs(user_id, parents_ids):

    if user_id == -1:
        print("ERROR")
        return None
    stopwatch = time.time()

    cppnState = CPPNState.objects.get(id=user_id)
    state = _pickle_loads(cppnState.data)

    pipeline = _get_pipeline()

    # render layouts

    print("Time elapsed 1: ", time.time() - stopwatch)

    selected_indices = pipeline.algorithm.select_winners(parents_ids)
    state = pipeline.evolve(state, selected_indices)

    population = pipeline.generate(state)
    pipeline.visualize_population(
        population, save_path="static/assets/generated", file_name=f"img_{user_id}"
    )

    cppnState.data = _pickle_dumps(state)
    cppnState.pub_date = timezone.now()
    cppnState.save()

    print("Time elapsed 2: ", time.time() - stopwatch)

    return list(range(IMAGES_PER_PAGE)), user_id


def _get_pipeline():
    global pipeline

    # check if pipeline exists
    if not "pipeline" in globals():
        print("setupping")
        test_genome = genome.DefaultGenome(
            num_inputs=2,
            num_outputs=4,
            node_gene=genome.DefaultNode(
                activation_options=[common.ACT.sigmoid, common.ACT.tanh, common.ACT.sin]
            ),
            init_hidden_layers=(2,)
        )

        algo = InteractiveNEAT(
            pop_size=IMAGES_PER_PAGE,
            genome=test_genome,
        )

        problem = InteractiveGrid(grid_size=(LAYOUT_RESOLUTION, LAYOUT_RESOLUTION))
        grid = plt.imread("aigs/images/cppn_inputs/piskel_example1.png")

        pipeline = InteractivePipeline(algorithm=algo, problem=problem, input_grid=grid)
    return pipeline


def _get_default_state():
    cppnstate0 = CPPNState.objects.filter(id=0)
    if len(cppnstate0) > 0 and os.path.exists("static/assets/generated/img_X_0.png"):
        return _pickle_loads(cppnstate0[0].data)

    pipeline = _get_pipeline()
    state = pipeline.setup()
    population = pipeline.generate(state)
    pipeline.visualize_population(
        population, save_path="static/assets/generated", file_name=f"img_X"
    )
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


def _get_new_user_id():
    default_state = _get_default_state()
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
        img_path = imgFromArray(
            pop[i], f"static/assets/generated/img_{layout.id}.png"
        )
        layout.img_path = img_path
        layout.save()


