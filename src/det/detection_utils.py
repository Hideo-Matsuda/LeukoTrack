import os
import sys
import tqdm
from matplotlib import pyplot as plt
import torch
import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.3
    
    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def plot(img, boxes, outpath, edgecolor='w'):
    fig, ax = plt.subplots(1, dpi=96)

    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    width, height, _ = img.shape
        
    ax.imshow(img, cmap='gray')
    fig.set_size_inches(width / 80, height / 80)

    for box in boxes:
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            linewidth=1.0, edgecolor=edgecolor)
        ax.add_patch(rect)

    plt.axis('off')
    try:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
    except FileNotFoundError:
        pass
    plt.savefig(outpath)
    plt.close()

def evaluate_and_write_result_files(model, data_loader, outdir=None, box_color='w', outfile=None):
    msg = 'Evaluating the model ...'
    if outdir is not None:
        msg = msg[:-3] + 'and saving the results to {} ...'.format(outdir) 
    print(msg)
    if outfile is not None:
        try:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, 'a') as f:
                print(msg, file=f)
        except:
            pass

    model.eval()
    results = {}
    for imgs, targets in tqdm.tqdm(data_loader):
        imgs = [img.to(next(model.parameters()).device) for img in imgs]

        with torch.no_grad():
            preds = model(imgs)

        for pred, target in zip(preds, targets):
            results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(),
                                                'scores': pred['scores'].cpu()}

                                            
    data_loader.dataset.print_eval(results, outfile=outfile)
    if outdir is not None:
        print('Visualizing ...')
        try:
            os.makedirs(os.path.dirname(outdir), exist_ok=True)
        except FileNotFoundError: # if dirname of outdir is ''
            pass
        data_loader.dataset.write_results_files(results, outdir)

        for data_id in tqdm.tqdm(sorted(results.keys())):
            img_path =  data_loader.dataset._img_paths[data_id]
            seq_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))  
            img_name = os.path.basename(img_path)
            img = data_loader.dataset[data_id][0]
            boxes = results[data_id]['boxes']
            outpath = os.path.join(outdir, 'images', seq_name, img_name)
            plot(img, boxes, outpath, edgecolor=box_color)

    print('Done.')
    if outfile is not None:
        try:
            with open(outfile, 'a') as f:
                print('Done', file=f)
        except:
            pass
