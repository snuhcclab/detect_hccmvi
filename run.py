# SNU HCC LAB Junseo Kang
# Run Experiment script; Build Model, Fit Model, Plot&Estimate Model
# python run.py raw_dir result_dir
import os.path
import shutil
import sys
from train.trainers import trainer, plotter
from train import experiments
import pdb


def main(raw_dir, result_dir):
    #
    #   Build Models
    #
    experiment_list = [
        experiments.CONRES3DMM
    ]

    #
    #   Prepare Directory
    #
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    #
    #   For Each Model
    #
    for experiment in experiment_list:
        os.mkdir(os.path.join(result_dir, experiment.name))
        os.mkdir(os.path.join(result_dir, experiment.name, 'logs'))
        os.mkdir(os.path.join(result_dir, experiment.name, 'plot'))
        os.mkdir(os.path.join(result_dir, experiment.name, 'model'))

        #
        #   Fit Model
        #
        experiment.model, history = trainer(
            experiment=experiment,
            result_dir=os.path.join(result_dir, experiment.name)
        )
        pdb.set_trace()
        #
        #   Plot Model
        #
        plotter(experiment=experiment,
                result_dir=os.path.join(result_dir, experiment.name),
                hist=history)


if __name__ == '__main__':
    main(raw_dir=sys.argv[1],
         result_dir=sys.argv[2])
