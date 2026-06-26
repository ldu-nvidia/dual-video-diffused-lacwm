from huggingface_hub import HfApi, login, snapshot_download


def download(repo_id, local_dir):
    login()
    downloaded_path = snapshot_download(
        repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False
    )

    print(f"Repository downloaded to: {downloaded_path}")


def upload_folder(folder_path, path_in_repo, repo_id="facebook/robotics-world-models"):
    api = HfApi()

    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
    )


def download_folder(repo_id, folder_name, local_dir):
    from huggingface_hub import login, snapshot_download

    login()
    # Download the entire repo
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        revision="main",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[
            f"{folder_name}/**"
        ],  # Only download contents matching this pattern
    )

    print(f"Repository folder downloaded to: {downloaded_path}")


if __name__ == "__main__":
    print(":)")
    # repo_id = "robotics-world-models"  # Replace with your repository ID
    # file_path_in_repo = "/enq_DINOWM_5Hz_p209_nodec/model_latest.pth"
    # # download("robotics-world-models/jepa-wm-rope", "/fsx-cortex/shared/robotics-world-models/models")
    # download_folder('facebook/robotics-world-models', 'jepa-wm-rope', '/home/abhagejji/eai/robot_world_models/model_registry/jepa/artem')

    # upload_folder(folder_path='/home/abhagejji/eai/robot_world_models/evaluation_tasks/mpk/push/v1/run_0001',
    #               path_in_repo='data/evaluation_tasks/mpk/push/v1/run_0001',
    #              )
    # upload_folder(folder_path='/home/abhagejji/eai/robot_world_models/evaluation_tasks/mpk/push/v2/run_0001',
    #               path_in_repo='data/evaluation_tasks/mpk/push/v2/run_0001',
    #              )
