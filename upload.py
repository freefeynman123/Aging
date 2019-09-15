import os

import click
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def listdir_nohidden(path):
    """
     List only non hidden files (useful in case when we want to ignore files/folders like __pychache__, .gitignore etc.)
    :param path: Path to files that we want to show.
    :return: List of files path with full directory.
    """
    for f in os.listdir(path):
        if not (f.startswith('.') or f.startswith('_')):
            f_out = os.path.join(path, f)
            yield f_out


@click.command()
@click.option('--clients_secrets_path', default='client_secrets.json',
              help='Path to the file with Google Drive secrets.')
@click.option('--files_folder', default='aging',
              help='Path to folder containing files that are going to be uploaded.')
@click.option('--creds_file', default='mycreds.txt',
              help='Name of the file containing credentials needed to authorize the app.')
@click.option('--notebook_training', default='Aging.ipynb',
              help='Name of the notebook with model training.')
def upload(clients_secrets_path, files_folder, creds_file, notebook_training):
    # Login to Google Drive and create drive object
    try:
        GoogleAuth.DEFAULT_SETTINGS["client_config_file"] = str(clients_secrets_path)
    except FileNotFoundError as e:
        print("Please provide correct name of secrets file.")
        raise e
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile(creds_file)
    if gauth.credentials is None:
        # Authenticate if they're not there
        # This is what solved the issues:
        gauth.GetFlow()
        gauth.flow.params.update({'access_type': 'offline'})
        gauth.flow.params.update({'approval_prompt': 'force'})
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()

    # Save the current credentials to a file
    gauth.SaveCredentialsFile(creds_file)
    drive = GoogleDrive(gauth)
    # Create folder on Google Drive to store files from given directory
    folder_metadata = {'title': files_folder, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    folderid = folder['id']
    for file in listdir_nohidden(files_folder):
        with open(file, "r") as f:
            fn = os.path.basename(f.name)
            file_drive = drive.CreateFile({'title': fn, "parents": [{"kind": "drive#fileLink", "id": folderid}]})
            file_drive.SetContentString(f.read())
            file_drive.Upload()
        print(f"The file: {fn} has been uploaded")
    file_train = drive.CreateFile({'title': notebook_training})
    file_train.SetContentFile(notebook_training)
    file_train.Upload()
    print(f"The file: {notebook_training} has been uploaded")


if __name__ == '__main__':
    upload()
