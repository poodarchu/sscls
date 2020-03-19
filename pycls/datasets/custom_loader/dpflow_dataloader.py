# encoding: utf-8
"""
@author: yuchen ma
@contact: mayuchen@megvii.com
@desc:
"""
import torch
import rrun
import uuid
from dpflow import InputPipe, OutputPipe, Controller, control
from .collate import numpy_collate


def _multiprocess_worker_loop(
    dataset, dataset_name, num_workers, batch_size, **kwargs
):

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=numpy_collate,
        num_workers=num_workers,
        batch_size=batch_size,
        **kwargs
    )

    output_pipe = OutputPipe(dataset_name)

    with control(io=[output_pipe]):
        while True:
            for minibatch in iter(data_loader):
                print(minibatch)
                output_pipe.put_pyobj(minibatch)


class DPFlowDataLoader:
    def __init__(
        self,
        dataset,
        dataset_name=None,
        batch_size=1,
        nr_gpu=1,
        dpflow_buffer_size=16,
        num_machines=1,
        num_workers=4,
        preemptible=False,
        **kwargs
    ):
        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )

        if num_machines < 0:
            raise ValueError(
                "num_machines option should be non-negative; "
                "use num_machines=0 to disable multiprocessing."
            )

        if batch_size % nr_gpu != 0:
            raise ValueError(
                "batch_size must be divisible by nr_gpu"
            )

        if dataset_name is None:
            self.dataset_name = str(uuid.uuid1())
        else:
            self.dataset_name = dataset_name

        self.dpflow_buffer_size = dpflow_buffer_size
        self.nr_gpu = nr_gpu
        self.batch_size = batch_size
        self.is_first = True
        self.num_machines = num_machines
        self.num_workers = num_workers
        self.dataset = dataset

        if num_workers >= 0 and num_machines > 0:
            self.spec = rrun.RunnerSpec()
            self.spec.name = "rrun-dataset-{}".format(self.dataset_name)
            self.spec.resources.cpu = self.num_workers
            self.spec.resources.memory_in_mb = 4096 * self.num_workers
            self.spec.priority = "Medium"
            if preemptible:
                self.spec.preemptible_flag = rrun.RunnerSpec.BestEffort
            else:
                self.spec.preemptible_flag = rrun.RunnerSpec.Unpreemptible

            self.spec.max_wait_time = 3600 * int(1e9)
            self.spec.minimum_lifetime = 24 * 3600 * int(1e9)
            self.spec.scheduling_hint.group = "users"
            self.spec.log_dir = '/data/outputs/pycls/imagenet'
            self._executor = rrun.RRunExecutor(self.spec, self.num_machines, 1)

            for i in range(self.num_machines):

                # _multiprocess_worker_loop(
                #     dataset=self.dataset,
                #     dataset_name=self.dataset_name,
                #     num_workers=self.num_workers,
                #     batch_size=self.batch_size // self.nr_gpu,
                #     **kwargs,
                # )

                self._executor.submit(
                    _multiprocess_worker_loop,
                    dataset=self.dataset,
                    dataset_name=self.dataset_name,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size // self.nr_gpu,
                    **kwargs,
                )

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        if self.is_first:
            self.input_pipe = InputPipe(
                self.dataset_name, buffer_size=self.dpflow_buffer_size
            )
            self.input_pipe.set_policy("GROUP_ID", self.dataset_name)
            self._worker_controller = Controller(io=[self.input_pipe])
            self._worker_controller.start()
            self.is_first = False

        for idx in range(len(self)):
            data_dict = self.input_pipe.get()
            yield data_dict
