'''
for file in os.listdir(PATH+"\\Original"):
    path = PATH + "\\Original\\" + file
    img = cv2.imread(path, 1)
    resized_img = cv2.resize(img, (96, 128), interpolation=cv2.INTER_CUBIC)

    X.append(resized_img)

    path2 = PATH + "\\Ground Truth\\" + file[:-4]+"_mask.png"
    msk = cv2.imread(path2, 0)

    resized_msk = cv2.resize(msk, (96, 128), interpolation=cv2.INTER_CUBIC)

    Y.append(resized_msk)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))
X_train = X_train / 255
X_test = X_test / 255
Y_train = Y_train / 255
Y_test = Y_test / 255
Y_train = np.round(Y_train,0)
Y_test = np.round(Y_test,0)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)



def evaluateModel(model, X_test, Y_test, batchSize,epoch):

    try:
        os.makedirs('results')
    except:
        pass

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)

    try:
        os.makedirs(f'results\\images\\epoch_{epoch+1:02d}')
    except:
        pass

    for i in range(10):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jacard_viz = ((np.sum(intersection)+1.0)/(np.sum(union)+1.0))
        plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard_viz))

        plt.savefig(f'results\\images\\epoch_{epoch+1:02d}\\i'+str(i)+'.png',format='png')
        plt.close()

    jacard_1 = 0
    jacard_2 = 0
    dice_1 = 0
    dice_2 = 0
    tversky_value=0
    smooth = 1.0
    alpha = 0.75

    for i in range(len(Y_test)):

        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()

        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection
        jacard_1 += ((np.sum(intersection)+smooth)/(np.sum(union)+smooth))

        true_pos = K.sum(y2 * yp_2)
        false_neg = K.sum(y2 * (1 - yp_2))
        false_pos = K.sum((1 - y2) * yp_2)

        tversky_value += (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

        dice_1 += (2. * np.sum(intersection) + smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)

        jacard_2 += jacard(Y_test[i].astype('float32'),yp[i].astype('float32'))
        dice_2 += dice_coef(Y_test[i].astype('float32'),yp[i].astype('float32'))

    jacard_1 /= len(Y_test)
    dice_1 /= len(Y_test)
    jacard_2 /= len(Y_test)
    dice_2 /= len(Y_test)
    tversky_value /= len(Y_test)



    print('Jacard Index : '+str(jacard_1))
    print('Dice Coefficient : '+str(dice_1))

    fp = open('results/log.txt','a')
    fp.write(str(epoch+1)+','+str(jacard_1)+','+str(dice_1)+','+str(tversky_value.numpy())+','
             +str(jacard_2.numpy()) + ',' + str(dice_2.numpy())+'\n')
    fp.close()

    fp = open('results/best.txt','r')
    best = fp.read()
    fp.close()

    if(jacard_1>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard_1))
        print('***********************************************')
        fp = open('results/best.txt','w')
        fp.write(str(jacard_1))
        fp.close()
        model.save(f'results\\best.h5')

    saveModel(model,epoch=epoch)

def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize,checkpoint_epoch=0):


    for epoch in range(checkpoint_epoch,epochs):
        print('Epoch : {}'.format(epoch+1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1,callbacks=callbacks_list)

        evaluateModel(model,X_test, Y_test,batchSize,epoch)

    return model

'''