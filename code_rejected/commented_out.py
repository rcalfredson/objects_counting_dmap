#*
# from data_loader.py
#*

# img = cv2.resize(img, (0, 0), fx=0.05, fy=0.05)
# reshape = img.reshape(
#     (img.shape[0] * img.shape[1], 3))
    
# cluster = KMeans(n_clusters=1).fit(reshape)

# return tuple([int(255*val) for val in cluster.cluster_centers_[0]])

# this code is still doing the same as before, is it not? It's just sampling one patch
# from each source image in the batch.
# patch_size = 160

# allowed range:
# say image were 200 pixels. lower bound can be from 0 to 199 - 160 = 139.
# a patch from 139 to 199 actually has 161 elements, though.

# print('skipping an empty.')
# img_patch = el[0][:, y_origin:y_origin+patch_size, x_origin:x_origin + patch_size]
# cv2.imshow('debug', img_patch.T)
# cv2.waitKey(0)

# print('size of original label:', el[1].shape)
# print('label patch size:', label_patch.size)

# print('values of the image patch:', img_patch[:, 0:100, 0:100])
# print('final image patch shape:', img_patch.shape)
# print(np.count_nonzero(img_patch))

# if numPatches > 1:
#     for i in range(resizedImgs.shape[0]):
#         cv2.imshow('debug', resizedImgs[i, :].T.numpy().astype(np.float32))
#         cv2.imshow('debug2', resizedLabels[i, :].T.numpy().astype(np.float32))
#         cv2.waitKey(0)

# resizedImgs, resizedLabels = [], []

# padded = torch.mul(el[0], 255)
# padded = transforms.ToPILImage()(padded.T)

# padded = funcTrans.pad(padded, padding, bckgnd, 'constant')
# print('shape before adding padding:', padded.shape)
# print('shape after transpose:', padded.T.shape)
# print('padding:', padding)
# print('maxHt and maxWidth:', maxHt, maxWidth)

# print('shape after adding padding:', padded.shape)
# resizedImgs.append(np.divide(np.array(padded).astype(np.float32), 255))

# padded = transforms.ToPILImage()(el[1].T)
# padded = funcTrans.pad(padded, padding, 0, 'constant')

# resizedLabels.append(np.expand_dims(np.array(padded), 0))
# resizedLabels.append(np.array(padded))

# preConversionTime = timeit.default_timer()
# print('shapes and types of resized images:')
# for img in resizedImgs:
#     print(img.shape)
#     print(type(img))
# print('shapes of types of resized labels:')
# for label in resizedLabels:
#     print(label.shape)
#     print(type(label))
# imgTensor, labelTensor = torch.FloatTensor(resizedImgs), torch.FloatTensor(resizedLabels)
# postConversionTime = timeit.default_timer() - preConversionTime
# print('proportion of padding dims calc time to total:', postConversionTime / overallTotalTime)
# print('shapes of final output tensors:', resizedImgs.shape, resizedLabels.shape)
# print('content of one pair of resized images and labels:', resizedImgs[1], resizedLabels[1])

#*
# from looper.py
#*

# print('value of self.dropout:', self.dropout)
# print('result before scale-down', result)
# if not self.validation and self.dropout:
    # label = torch.mul(label, 10*DROPOUT_PROB)
# print('result after scale-down:', result)
# print('result:', result)
# print('label:', label)

counter = 0
# countLoss = torch.zeros(label.shape[0]).cuda()
# print('shape of input:', image.shape)
# print('shape of result:', result.shape)
# print('shape of label:', label.shape)

# countLoss[counter] = np.power(true_counts - predicted_counts, 2.0)
# counter += 1
# if counter < 4:
#     cv2.imwrite('debug_imgs/orig_img_%i.png'%counter, 255*image.cpu().numpy()[counter].T)
#     cv2.imwrite('debug_imgs/orig_img_%i.png'%counter, 255*cv2.cvtColor(image.cpu().numpy()[counter].T, cv2.COLOR_BGR2RGB))
#     dMapToShow = predicted.cpu().detach().numpy()[0].T
#     cv2.imwrite('debug_imgs/predictions_%i.png'%counter, 255*dMapToShow)
#     cv2.imwrite('debug_imgs/labelsOrig_%i.png'%counter, 255*label.cpu().detach().numpy()[counter].T)
#     counter += 1
    
#     cv2.waitKey(0)
#     counter += 1
    # labelAsNP = label.cpu().detach().numpy()[0].T
#     print('max value in labels:', np.amax(labelAsNP))
    # cv2.imshow('originalLabels:', labelAsNP)
#     plt.figure()
#     plt.imshow(np.squeeze(true.cpu().detach().numpy().T))
#     plt.figure()
#     plt.imshow(np.squeeze(predicted.cpu().detach().numpy().T))
#     plt.show()
        # print('true counts:', true_counts)
        # print('original size of labels:', label.shape)
        # print('predicted vals:', predicted_counts)
        # temporarily resize for debugging
    # input('finished writing images')
    # print('loss:', loss)
    # print('true values:', self.true_values)
    # print('predicted values:', self.predicted_values)
    # input('enter..')
    # print('shape of result:', result.shape)

# determine amount of padding by comparison with the outputted density map.
asymmetryCorrs = {'v': 0, 'h': 0}
# print('horiz excess before int?', (label.shape[-1] - result.shape[-1]) / 2)
hDiff = label.shape[-1] - result.shape[-1]
# horizExcess = int(hDiff / 2)
if hDiff % 2 > 0:
    asymmetryCorrs['h'] = 1
else:
    asymmetryCorrs['h'] = 0
# print('vert excess before int?', (label.shape[-2] - result.shape[-2]) / 2)
vDiff = label.shape[-2] - result.shape[-2]
# vertExcess = int(vDiff / 2)
if vDiff % 2 > 0:
    asymmetryCorrs['v'] = 1
else:
    asymmetryCorrs['v'] = 0

label = label[:, :, :label.shape[-2] - vDiff,
    :label.shape[-1] - hDiff]
loss = self.loss(result, label)
self.running_loss[-1] += image.shape[0] * loss.item() / self.size
# print('label shape before resize?', label.shape)
# labelsOldStyle = torch.nn.functional.interpolate(label, result.shape[-2:])
# label = label[:, :, vertExcess + asymmetryCorrs['v']: label.shape[-2] - vertExcess,
    # horizExcess + asymmetryCorrs['h']:label.shape[-1] - horizExcess]

# print('amount removed from top:',  vertExcess + asymmetryCorrs['v'])
# print('amount removed from bottom:', vertExcess)
# print('amount removed from left:', horizExcess + asymmetryCorrs['h'])
# print('from right:', horizExcess)
# print('horiz access?', horizExcess)
# print('vert access?', vertExcess)
# print('asymmetry?', asymmetryCorrs)
# print('label shape after resize?', label.shape)
# for i in range(len(label)):
#     dMapToShow = result.cpu().detach().numpy()[i].T
#     oldLabel = labelsOldStyle.cpu().detach().numpy()[i].T
#     labelAsNP = label.cpu().detach().numpy()[i].T
#     print('shape of old labels:', oldLabel.shape)
#     cv2.imwrite('debug_imgs/labelsResized_%i.png'%i, 255*labelAsNP)
#     cv2.imwrite('debug_imgs/labelsResizedOldStyle_%i.png'%i, 255*oldLabel)
# input('finished writing batch. Enter to continue.')

