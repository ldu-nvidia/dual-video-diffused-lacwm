import torch


def get_profiler(wait=1, warmup=5, active=3, repeat=1, directory="./profiler_logs"):
    total_steps = repeat * (wait + warmup + active)
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(directory),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    return profiler, total_steps


if __name__ == "__main__":
    # Example usage
    profiler, total_steps = get_profiler(directory="./tmp/profiler_logs")
    profiler.start()

    # do some random torch operations
    for _ in range(total_steps):
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        profiler.step()

    # When done, stop the profiler -- it will save the trace to the specified directory
    profiler.stop()
