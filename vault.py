import os
import pathlib


# CREATE VAULT INSTANCE CLASS #

class VaultClient:
    # initialize empty dictionary to store secrets in it
    _secrets: dict = {}

    def __init__(self):
        # read mounted secret files from given directory
        self._secrets = self.load_vault_secrets(directory="/vault/secrets")
        # count number of existing secrets
        count = len(self._secrets)
        print(f"Loaded {count} secrets from vault")

    def get(self, key: str) -> str:
        # returns the value of the item with the specified key
        return self._secrets.get(key)

    def load_vault_secrets(self, directory: str) -> dict:
        # initialize empty dictionary to store secrets in it
        out = {}

        # return nothing if no directory was specified
        if not directory:
            return out

        # iterate over existing paths
        for path in pathlib.Path(directory).rglob('*'):
            # skip current iteration if no file was found in path
            if not path.is_file():
                continue
            # read the file located in the path
            secrets_dic, err = self.read_file(path)
            # skip current iteration if error was raised while searching for file in the path
            if err:
                print(f"Error reading vault file {path}: {err}")
                continue
            # store secrets in dictionary
            for k, v in secrets_dic.items():
                out[k] = v

        return out

    def read_file(self, path: str) -> (dict, Exception):
        # initialize empty dictionary to store files in it
        secrets_dic = {}

        # try iterating over existing lines (key,value) in file
        try:
            with open(path, 'r') as file:
                # read lines
                for line in file:
                    line = line.strip()
                    # skip current iteration if no line was found in file
                    if not line:
                        continue
                    # split key,value
                    parts = line.split(': ')
                    # validate line configuration
                    if len(parts) < 2:
                        # if ':' in line:
                        #     continue
                        return None, ValueError(f"invalid configuration line {line}")
                    # store key,value
                    secrets_dic[parts[0]] = parts[1]
        # raise exception we meant to catch from try
        except Exception as e:
            return None, e

        return secrets_dic, None


# INITIALIZE VAULT PROCESS #

def get_vault_secrets():
    # initialize Vault secrets retrieval process
    vaultClient = VaultClient()

    # retrieve redshift secrets
    redshift_credentials = {"host": vaultClient.get("REDSHIFT_HOST"),
                            "port": vaultClient.get("REDSHIFT_PORT"),
                            "dbname": vaultClient.get("REDSHIFT_DBNAME"),
                            "user": vaultClient.get("REDSHIFT_USER"),
                            "password": vaultClient.get("REDSHIFT_PASSWORD")
                            }

    # retrieve YouTube secrets
    youtube_api = {"API_KEY": vaultClient.get("YOUTUBE_API")}

    return redshift_credentials, youtube_api
