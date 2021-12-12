#! /usr/bin/env python3

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

CACHE_ITEM_SIZE_BYTES = 5 * 1024

###################################
# Parse Args, Set paramaters
###################################
base_parameters = {}
base_parameters["experiment_name"] = "One Distance ORACLE PTN"
base_parameters["lr"] = 0.001
base_parameters["device"] = "cuda"
base_parameters["max_cache_items"] = 4.5e6

base_parameters["seed"] = 1337
base_parameters["desired_serial_numbers"] = ALL_SERIAL_NUMBERS
# base_parameters["desired_serial_numbers"] = [
#     "3123D52",
#     "3123D65",
#     "3123D79",
#     "3123D80",
# ]
base_parameters["source_domains"] = [38,]
base_parameters["target_domains"] = [20,44,
    2,
    8,
    14,
    26,
    32,
    50,
    56,
    62
]

base_parameters["window_stride"]=50
base_parameters["window_length"]=256
base_parameters["desired_runs"]=[1]
base_parameters["num_examples_per_device"]=75000

base_parameters["n_shot"] = 10
base_parameters["n_way"]  = len(base_parameters["desired_serial_numbers"])
base_parameters["n_query"]  = 10
base_parameters["n_train_tasks"] = 2000
base_parameters["n_train_tasks"] = 100
base_parameters["n_val_tasks"]  = 100
base_parameters["n_test_tasks"]  = 100

base_parameters["n_epoch"] = 100
base_parameters["n_epoch"] = 3

base_parameters["patience"] = 10


base_parameters["x_net"] =     [# droupout, groups, 512 out
    {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 128]}},
    {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":(1,7), "bias":False, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":256}},

    {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":(2,7), "bias":True, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":80}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 80*128, "out_features": 256}}, # 80 units per IQ pair
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm1d", "kargs": {"num_features":256}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
]



def metrics_calculator(parameters):
    def n_val_tasks_to_saturate_val(
        n_serials,
        n_examples_per_device,
        n_way,
        n_shot,
        n_query,
        val_split_size
    ):
        return (n_serials * n_examples_per_device * val_split_size) / \
            n_example_used_in_each_episode(n_way, n_shot, n_query)["total"]


    def n_example_used_in_each_episode(
        n_way,
        n_shot,
        n_query,
    ):
        s = n_way * n_shot
        q = n_way * n_query

        return {
            "query": q,
            "support": s,
            "total": q+s
        }


    def n_examples_per_device_and_distance(
        n_examples_per_device,
        n_source_domains,
        n_target_domains,
    ):
        return {
            "source": n_examples_per_device/n_source_domains,
            "target": n_examples_per_device/n_target_domains,
        }


    def n_examples(
        n_serials,
        n_examples_per_device,
    ):
        return {
            "source": n_serials * n_examples_per_device,
            "target": n_serials * n_examples_per_device,
            "total": 2*(n_serials * n_examples_per_device)
        }

    def max_cache_size(
        n_cache_items,
        n_serials,
        n_examples_per_device,
    ):
        return max(
            n_cache_items * CACHE_ITEM_SIZE_BYTES,
            n_examples(n_serials, n_examples_per_device)["total"]
        )
    ########################################################################
    n_examples_per_device=parameters["num_examples_per_device"]
    n_query=parameters["n_query"]
    n_serials=len(parameters["desired_serial_numbers"])
    n_shot=parameters["n_shot"]
    n_source_domains=len(parameters["source_domains"])
    n_target_domains=len(parameters["target_domains"])
    n_way=parameters["n_way"]
    val_split_size=0.15
    n_cache_items = parameters["max_cache_items"]

    print("======================================================================================")
    print("Absolute minimum num val episodes to saturate validation set:", 
        n_val_tasks_to_saturate_val(
            n_serials,
            n_examples_per_device,
            n_way,
            n_shot,
            n_query,
            val_split_size
        )
    )

    print("======================================================================================")
    ligma = n_example_used_in_each_episode(
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
    )
    print("Num of examples used in each episode:")
    print("  support:", ligma["support"])
    print("  query:", ligma["query"])
    print("  total:", ligma["total"])

    print("======================================================================================")
    sugma = n_examples_per_device_and_distance(
        n_examples_per_device=n_examples_per_device,
        n_source_domains=n_source_domains,
        n_target_domains=n_target_domains
    )
    print("Number of examples per (device,distance):")
    print("  source:", sugma["source"])
    print("  target:", sugma["target"])

    print("======================================================================================")
    fugma = n_examples(
        n_serials=n_serials,
        n_examples_per_device=n_examples_per_device
    )
    print("Number of examples used in the system:")
    print("  source:", fugma["source"])
    print("  target:", fugma["target"])
    print("  total:", fugma["total"])

    print("======================================================================================")
    print("Maximum cache size: {:.1f}GiB".format( 
        max_cache_size(n_cache_items=n_cache_items, n_serials=n_serials, n_examples_per_device=n_examples_per_device)/1024/1024/1024
    ))

    print("======================================================================================")


metrics_calculator(base_parameters)

