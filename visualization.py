from pickletools import optimize
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch 
from PIL import Image
import antialiased_cnns
import torchvision.transforms as transforms
from save_torch_tensor import create_val_loader
from torchvision import models
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from models.cifar import resnet_3D, resnet_group

def val_transform():
    transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])
    return transform

def get_model( model_name, dict_path=None ):
    if dict_path == None:
        print( "Not loading checkpoint" )

    if model_name == 'resnet' or model_name == 'origin':
        model = getattr(models, "resnet18")(pretrained=False)
    elif model_name == '3D' or model_name == '3d':
        print('laod 3D model')
        model = resnet_3D.ResNet18(num_classes=1000)
    elif model_name == 'group':
        model = resnet_group.ResNet18(num_classes=1000)
    elif model_name == 'aacnn':
        model = antialiased_cnns.resnet18()
    else:
        print("invalid model name")
        raise NotImplementedError

    if dict_path != None:
        print( f"loading checkpoint from {dict_path}" )

        state_dict = torch.load( dict_path )
        new_dict = {}
        for key in state_dict.keys():
            new_key = key[7:]
            new_dict[new_key] = state_dict[key]
        model.load_state_dict( new_dict )
        print( "checkpoint loaded" )


    model = model.cuda()
    return model

def shift_img(batch_tensor):
    output_tensor = []
    for i in range(8):
        for j in range(8):
            output_tensor.append( batch_tensor[ :, 7-i:, 7-j: ] ) 
    
    return output_tensor

def produce_tensoe():
    val_dataset = "/data/datasets/ntsai/val_400_0.1_90.ffcv"
    val_loader = create_val_loader(val_dataset, 16, 128, 256, 0, preprocess = True )
    
    model = getattr(models, "resnet18")(pretrained=False)
    state_dict = torch.load( './imagenet_original.pt' )

    new_dict = {}
    for key in state_dict.keys():
        new_key = key[7:]
        new_dict[new_key] = state_dict[key]

    model.load_state_dict( new_dict )
    model = model.cuda()
    i = 0
    with torch.no_grad():
        with autocast():
            with tqdm(val_loader, unit="batch") as tepoch:
                for images, target in tepoch:
                    if i == 10:
                        break
                    # images = images.half() 
                    # torch.save( images, f"./val_images/val_processed_image_batch_{i}.pt" )
                    # torch.save( target, f"./val_images/val_label_batch_{i}.pt" )
                    # output = model(images)
                    # torch.save( output, f"./val_images/val_predc_batch_{i}.pt" )
                    i += 1
                

def box_plot(data, out_name, model_name = 'Baseline', ): 

    all_boxes = []
    B = plt.boxplot(data)
    v = [item.get_ydata() for item in B['whiskers']]
    tile1 = v[0][0]
    min_val = min(data)
    max_val = max(data)

    data = [0] + list(data) # offset, the first value always plotted wrong.
    for i, data_point in enumerate(data):
        fig = plt.figure()
        axs = fig.subplots(1)
        axs.set_ylim([0, 100])
        axs = sns.set(rc={'figure.figsize':(3, 10)})

        axs = sns.boxplot( data=data, showfliers = True, width = [0.125], color = 'gray' )

        if data_point > tile1:
            axs.bar('v', data_point, color = '#7FFF00', width = 0.25, alpha = 0.9)
        elif( data_point < min_val+0.1 ):
            axs.bar('v', data_point, color = '#FF0000', width = 0.25, alpha = 0.9)            
        else:
            axs.bar('v', data_point, color = '#FFFF14', width = 0.25, alpha = 0.9)

        plt.title( str(round(data_point, 2)) )
        plt.text(-0.06, -9, model_name, fontsize=18)

        # ax.figure.savefig("sns-heatmap.png")
        fig.canvas.draw()
        box_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        box_plot = box_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        box_plot = Image.fromarray(box_plot)
        all_boxes.append(box_plot)

        box_plot.save(f"./heatmap_figs/box_{i}.png")

    make_gif( all_boxes, out_name )

def make_gif( boxes, out_name ):
    print(len(boxes))
    boxes[1].save( out_name, save_all=True, duration = 250, append_images=boxes[2:], loop=0, optimize = False )
    # imageio.mimsave('out_name', boxes[1:])


def prediction(model, image_stack):
    tensor_transform = val_transform()
    tensor_stack = []
    for img in image_stack:
        tensor = tensor_transform( img ).reshape( ( 3, 224, 224) )
        tensor_stack.append(tensor)
    tensor_stack = torch.stack( tensor_stack  ).cuda()
    prediction = model( tensor_stack )
    confidence = torch.nn.Softmax(dim=1)(prediction)
    # confidence = torch.nn.Sigmoid()(prediction)
    # print( confidence.shape )
    return confidence

'''
real_image_path = "./val_images/val_image_batch_0.pt"
img_gif_path =  'shifted_img.gif'
model_type = '3d'
model_check_point_path = './imagenet_3D_trained.pt' 
real_label_path = "./val_images/val_label_batch_0.pt"
box_plot_gif_path = 'example.gif'
box_plot_title = '"3d conv"'
'''
def AA_visualization( 
    idx = 28,
    model = None,
    real_image_path = "./val_images/val_image_batch_0.pt",
    img_gif_path = None,
    real_label_path = "./val_images/val_label_batch_0.pt", 
    box_plot_gif_path = 'boxplot_aacnn.gif',
    box_plot_title = 'aacnn'   ):


    real_images = torch.load(real_image_path) 
    img = real_images[idx] # type(image) = PIL image
    shifted_images = shift_img(img)

    T = transforms.ToPILImage()
    image_stack = [T(img)] # 1 offset for making gif
    for i, img in enumerate(shifted_images):
        img = T(img)
        image_stack.append( img )
        # img.save( f"./heatmap_figs/shift_{i}.png" )

    if img_gif_path is not None:
        make_gif( image_stack, img_gif_path)

    confidence = prediction(model, image_stack[1:] )
    

    real_label = torch.load( real_label_path )
    label = real_label[idx].item()
    confidence = confidence[:, label]
    confidence = confidence.cpu().detach().numpy()*100
    print(confidence)
    box_plot(confidence, box_plot_gif_path, box_plot_title)

def main():
    aacnn_model = get_model( 'aacnn', dict_path = './imagenet_aacnn.pt'  )
    aacnn_model = aacnn_model.eval()

    D3_model = get_model( '3D', dict_path = './imagenet_3D_trained.pt'  )
    D3_model = D3_model.eval()

        
    for idx in range(17):
        print(idx)
        print( 'aacnn conf' )
        AA_visualization(idx = idx, 
                        model = aacnn_model,
                        img_gif_path = None,
                        img_gif_path = f'shifted_gif/shifted_img_{idx}.gif',
                        box_plot_gif_path = f'box_plot/boxplot_aacnn_{idx}.gif',
                        box_plot_title = 'aacnn' )

        print( '3D conv conf' )
        AA_visualization(idx = idx, 
                        model = D3_model,
                        box_plot_gif_path = f'box_plot/boxplot__3D_conv_{idx}.gif',
                        box_plot_title = '3D_conv' )


if __name__ == '__main__':
    main()