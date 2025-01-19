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

from .models import Layout

from .tools.SGAA_test import init_population

from polls.tools.SimpleGAArrays import GenerateNewPopulation
from polls.tools.SGAA_test import differentTest, imgFromArray


def IndexView(request):
    template_name = "index.html"
    images = 9
    n = list(map(str,range(1,10)))
    context = { 'images': images, 'n': n }

    # check if default exists
    a = Layout.objects.filter(id__in=n).values_list('data', flat=True)
    if(len(a) < 9):
        for offspring in init_population():
            layout = Layout(data=json.dumps(offspring.tolist()), pub_date=timezone.now())
            layout.save()
            img_path = imgFromArray(offspring, f"static/assets/generated/img_{layout.id}.png")
            layout.img_path = img_path
            layout.save()

    return render(request, template_name, context)

def process_images(request):
    if request.method == 'POST':
        data = json.loads(request.body)  # Parse the incoming JSON
        #print(data)
        #selected_images = data.get('selected_images', [])
        #all_images = data.get('all_images', [])
        #
        # Here, you can process the selected images further (e.g., save to DB, perform actions, etc.)
        #new_images = []
        #for i in all_images:
        #    if i in selected_images:
        #        new_images.append(i)
        #    else:
        #        number_to_add = random.randint(0,13)
        #        while str(number_to_add) in new_images or str(number_to_add) in selected_images:
        #            number_to_add = (number_to_add + 1) % 14
        #        new_images.append(str(number_to_add))

        parents_ids = data.get('selected_images', [])
        all_ids = data.get('all_images', [])

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




        return JsonResponse({'message': 'Images processed successfully', 'new_images': new_ids})

    return JsonResponse({'error': 'Invalid request'}, status=400)