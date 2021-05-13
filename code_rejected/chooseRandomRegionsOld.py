def chooseRandomRegions(batch, numPatches, shuffle=False, skipEmpties=False):
    # this code is still doing the same as before, is it not? It's just sampling one patch
    # from each source image in the batch.
    patch_size = 160
    resizedImgs = torch.zeros((len(batch)*numPatches, 3, patch_size, patch_size), dtype=torch.float32)
    resizedLabels = torch.zeros((len(batch)*numPatches, 1, patch_size, patch_size), dtype=torch.float32)
    for i, el in enumerate(batch):
        # allowed range:
        # say image were 200 pixels. lower bound can be from 0 to 199 - 160 = 139.
        # a patch from 139 to 199 actually has 161 elements, though.
        if skipEmpties and np.count_nonzero(el[1]) == 0:
            print('skipping an empty.')
            # input('enter...')
            continue
        counter = 0
        while counter < numPatches:
            x_origin = np.random.randint(0, el[0].shape[-1] - patch_size)
            y_origin = np.random.randint(0, el[0].shape[1] - patch_size)
            label_patch = el[1][:, y_origin:y_origin+patch_size, x_origin:x_origin + patch_size]
            if skipEmpties and np.count_nonzero(label_patch) == 0:
                # print('skipping an empty.')
                # img_patch = el[0][:, y_origin:y_origin+patch_size, x_origin:x_origin + patch_size]
                # cv2.imshow('debug', img_patch.T)
                # cv2.waitKey(0)
                continue
            img_patch = el[0][:, y_origin:y_origin+patch_size, x_origin:x_origin + patch_size]
            # print('size of original label:', el[1].shape)
            # print('label patch size:', label_patch.size)
            img_patch, label_patch = rotateInCollateStep(img_patch, label_patch)
            # print('values of the image patch:', img_patch[:, 0:100, 0:100])
            # print('final image patch shape:', img_patch.shape)
            # print(np.count_nonzero(img_patch))
            resizedImgs[i*numPatches + counter] = torch.from_numpy(img_patch)
            resizedLabels[i*numPatches + counter] = torch.from_numpy(label_patch.T)
            counter += 1
    if shuffle:
        indexOrder = list(range(resizedImgs.shape[0]))
        random.shuffle(indexOrder)
        indexOrder = torch.LongTensor(indexOrder)
        resizedImgsOld = resizedImgs
        resizedLabelsOld = resizedLabels
        resizedImgs = torch.zeros_like(resizedImgsOld)
        resizedLabels = torch.zeros_like(resizedLabelsOld)
        resizedImgs = resizedImgsOld[indexOrder]
        resizedLabels = resizedLabelsOld[indexOrder]
    # if numPatches > 1:
    #     for i in range(resizedImgs.shape[0]):
    #         cv2.imshow('debug', resizedImgs[i, :].T.numpy().astype(np.float32))
    #         cv2.imshow('debug2', resizedLabels[i, :].T.numpy().astype(np.float32))
    #         cv2.waitKey(0)
    return resizedImgs, resizedLabels