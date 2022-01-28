import torch
from tqdm import tqdm

def evaluation(model, dataloaders , project):
    running_corrects =0 
    print("")
    results = []
    for i in tqdm(range(len(dataloaders['val'].dataset) )):
        inputs, labels = next(iter(dataloaders['val']))
        outputs = model(inputs).data
        _, preds = torch.max(outputs, 1)
        results.append([outputs , preds])
        running_corrects += torch.sum(preds == labels.data)
        #print(running_corrects )

    accuracy = running_corrects.double() / len(dataloaders['val'].dataset)
    print("Accuracy of the CNN: " , accuracy)
    return running_corrects , results, accuracy