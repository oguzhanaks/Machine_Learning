import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import numpy as np
from matplotlib import pyplot as plt





if __name__ == '__main__': 
    print(torch.cuda.is_available())   
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #cuda gpu kullanım 
    

    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            # VGG de bulunan imageler 224 e 224 olduğu için tekrar boyutlandırıyoruz
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #normalizasyon
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    data_dir = 'yuz - Kopya'        #veri yolu
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=16,
                                                 shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=16,
                                                 shuffle=False, num_workers=4)
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    for train in train_loader:
        
        break
    
    def imshow(inp, title=None):
        #görselleştirme işlemleri
       
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
    
    # Train verilerini batch e gönderdik
    inputs, classes = next(iter(train_loader))
    
    
    out = torchvision.utils.make_grid(inputs)
    
    imshow(out, title=[class_names[x] for x in classes])
    
    net = models.vgg16(pretrained=True)
    net = net.to(device)
    # vgg16 eğitilmiş model yükleme
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    net
    
    for param in net.parameters(): # Eğitim setinin tüm katmanları dondurduk
        param.requires_grad = False
    net = net.to(device)
    
    num_ftrs = net.classifier[6].in_features # VGG 16da ki 6 katmanlı diziye erişiyoruz.
    net.classifier[6] = nn.Linear(num_ftrs, 2).to(device) 
    net
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params= net.parameters(), lr=0.01,momentum=0.8) #öğrenme oranı 0.01

    
    num_epochs = 25
    
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        
        #train
        net.train()
        for i, (images, labels) in enumerate(train_loader):
          images, labels = images.to(device), labels.to(device)
          
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)
          train_loss += loss.item()
          train_acc += (outputs.max(1)[1] == labels).sum().item()
          loss.backward()
          optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)
        #print ('Epoch [{}/{}], Loss: {loss:.4f}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, lr：{learning_rate}' 
                          # .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, train_loss=avg_train_loss, train_acc=avg_train_acc, learning_rate=optimizer.param_groups[0]["lr"]))
        #val
        net.eval()
        with torch.no_grad():
          for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)
        
        print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, lr：{learning_rate}' 
                           .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc, learning_rate=optimizer.param_groups[0]["lr"]))
     
        lr_scheduler.step()
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)
        
    plt.figure()
    plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='eğitim_doğruluğu')
    plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='doğrulama_doğruluğu')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('doğruluk')
    plt.title('Eğitim ve Doğrulama doğrluğu')
    plt.grid()        
   
    
    plt.figure()
    plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='eğitim_kayıp')
    plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='doğrulama_kayıp')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('kayıp')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.grid()
    

    
    
    
  
                
   
        
                 

    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        
    
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
    
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('tahmin: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])
    
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
     
    visualize_model(net)
    
