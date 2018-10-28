
class Arguments:

    def __init__(self):
        self.name = "Arguments"
        self.size_image = (224,224,3) # On utilise des images de 160 x 160 pixels, en couleurs. (p-e passage grayscale?)
        self.n_classes = 2 #Murs ou non
        self.activation_output = 'softmax'
        self.dropout_value = 0.5
        self.activation_inside = 'relu'


def get_args():
    arg = Arguments()
    return arg
