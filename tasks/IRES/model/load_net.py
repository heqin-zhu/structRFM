from model.IreseekNet import IreeSeek_LM_model
                             
def get_model(MODEL_NAME, net_params):
    models = {
        'IreeSeek_LM_model': IreeSeek_LM_model
    }
        
    return models[MODEL_NAME](net_params)
