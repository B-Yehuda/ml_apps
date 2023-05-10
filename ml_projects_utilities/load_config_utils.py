import configparser
import os
import os.path


def load_config(model_names, model_types):
    # store config files dictionary
    config_objects = {}

    # populate dictionary with config files
    for m_name in model_names:
        config_objects[m_name] = []
        for m_type in model_types:
            # get config file name
            con_name = "config_" + m_type + "_" + m_name + ".ini"
            # initialize configparser object
            config = configparser.ConfigParser()
            # navigate to parent dir
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if not os.path.isdir(parent_dir):
                raise ValueError("Parent directory not found")
            # navigate to configs dir
            configs_dir = os.path.join(parent_dir, "configs")
            if not os.path.isdir(configs_dir):
                raise ValueError("Configs directory not found")
            # read from config file (location = parent_dir / configs)
            config.read(os.path.join(configs_dir, con_name))
            # store config file
            config_objects[m_name].append(config)

    return config_objects
