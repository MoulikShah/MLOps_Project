import ray
import subprocess

@ray.remote(num_gpus=1)
def train_with_shell():
    result = subprocess.run(["bash", "run.sh"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    return result.returncode

if __name__ == "__main__":
    ray.init(address="auto")
    result = ray.get(train_with_shell.remote())
    print(f"Training job exited with code {result}")
