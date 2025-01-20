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

from .tools.SGAA_test import init_population

from cwapp.tools.SimpleGAArrays import GenerateNewPopulation
from cwapp.tools.SGAA_test import differentTest, imgFromArray
from django.core.cache import cache


from aigs.interactive_NEAT import InteractiveNEAT
from aigs.interactive_problem import InteractiveGrid
from aigs.interactive_pipeline import InteractivePipeline
from tensorneat import genome, common
import matplotlib.pyplot as plt

import base64
import pickle



def IndexView(request,version="A"):
    template_name = "index.html"
    images = 9
    if version not in ["A","B"]:
        version = "A"
    n = list(map(str,range(1,10))) if version == "B" else list(map(str,range(9)))
    user_id = -1

    # check if default exists
    if version == "B":
        a = Layout.objects.filter(id__in=n).values_list('data', flat=True)
        if(len(a) < 9):
            for offspring in init_population():
                layout = Layout(data=json.dumps(offspring.tolist()), pub_date=timezone.now())
                layout.save()
                if version == "A":
                    img_path = imgFromArray(offspring, f"static/assets/generated/img_X_{layout.id}.png")
                else:
                    img_path = imgFromArray(offspring, f"static/assets/generated/img_{layout.id}.png")
                layout.img_path = img_path
                layout.save()
    else:
        user_id = _get_new_user_id()
        
    context = { 'images': images, 'n': n, 'version': version, 'user_id' : user_id}
    return render(request, template_name, context)

def process_images(request):
    if request.method == 'POST':
        data = json.loads(request.body)  # Parse the incoming JSON

        parents_ids = data.get('selected_images', [])
        all_ids = data.get('all_images', [])
        version = data.get('version', 'A')
        if version not in ["A","B"]:
            version = "A"

        if version == "B":
            new_ids = _nocppn_process_imgs(parents_ids,all_ids)
            user_id = -2
        else:
            user_id = data.get('user_id', -1)
            print("user_id: ", user_id)
            new_ids, user_id = _cppn_process_imgs(user_id, parents_ids)

        return JsonResponse({'message': 'Images processed successfully', 'new_images': new_ids, 'user_id': user_id})

    return JsonResponse({'error': 'Invalid request'}, status=400)


def _nocppn_process_imgs(parents_ids,all_ids):
    # get parents from database
    parents = Layout.objects.filter(id__in=parents_ids).values_list('data', flat=True)
    parents = [json.loads(p) for p in parents]

    # get new population
    offsprings = GenerateNewPopulation(parents, 9-len(parents), 0.1, 0.5,keep_parents=False)

    # add each new layout to the database and get id
    offsprings_ids = []
    for offspring in offsprings:
        layout = Layout(data=json.dumps(offspring.tolist()), pub_date=timezone.now())
        layout.save()
        offsprings_ids.append(layout.id)
        img_path = imgFromArray(offspring, f"static/assets/generated/img_{layout.id}.png")
        layout.img_path = img_path
        layout.save()


    new_ids = []
    for id in all_ids:
        if id in parents_ids:
            new_ids.append(id)
        else:
            new_ids.append(str(offsprings_ids.pop(0)))

    
    # remove old layouts
    delete_ids = [id for id in all_ids if id not in new_ids and int(id) > 9]
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
    state = pipeline.evole(state, selected_indices)
    
    state, population = pipeline.step(state)
    pipeline.visualize_population(population, save_path="static/assets/generated", file_name=f"img_{user_id}")

    cppnState.data = _pickle_dumps(state)
    cppnState.pub_date = timezone.now()
    cppnState.save()

    print("Time elapsed 2: ", time.time() - stopwatch)

    return list(range(9)), user_id
    
def _get_pipeline():
    global pipeline

    #check if pipeline exists
    if not 'pipeline' in globals():
        print("setupping")
        test_genome = genome.DefaultGenome(
            num_inputs=2,
            num_outputs=4,
            node_gene=genome.DefaultNode(
                activation_options=[common.ACT.sigmoid, common.ACT.tanh, common.ACT.sin]
            ),
        )

        algo = InteractiveNEAT(
            pop_size=9,
            genome=test_genome,
        )

        problem = InteractiveGrid(grid_size=(16, 16))
        grid = plt.imread("aigs/images/cppn_inputs/piskel_example1.png")

        pipeline = InteractivePipeline(algorithm=algo, problem=problem, input_grid=grid)
    return pipeline

def _get_default_state():
    cppnstate0 = CPPNState.objects.filter(id=0)
    if len(cppnstate0) > 0 and os.path.exists("static/assets/generated/img_X_0.png"):
        return _pickle_loads(cppnstate0[0].data)
    
    pipeline = _get_pipeline()
    state = pipeline.setup()
    state, population = pipeline.step(state)
    pipeline.visualize_population(population, save_path="static/assets/generated", file_name=f"img_X")
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
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')

def _get_new_user_id():
    default_state = _get_default_state()
    cppnState = CPPNState(data=_pickle_dumps(default_state),pub_date=timezone.now())
    cppnState.save()
    return cppnState.id
    
    