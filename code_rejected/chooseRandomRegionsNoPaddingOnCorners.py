def chooseRandomRegions(batch, numPatches, size, shuffle=False, skipEmpties=False):
    # this code is still doing the same as before, is it not? It's just sampling one patch
    # from each source image in the batch.
    resizedImgs = torch.zeros((len(batch)*numPatches, 3, size, size), dtype=torch.float32)
    resizedLabels = torch.zeros((len(batch)*numPatches, 1, size, size), dtype=torch.float32)
    for i, el in enumerate(batch):
        # allowed range:
        # say image were 200 pixels. lower bound can be from 0 to 199 - 160 = 139.
        # a patch from 139 to 199 actually has 161 elements, though.
        if skipEmpties and np.count_nonzero(el[1]) == 0:
            # input('enter...')
            continue
        # what is the shape when this gets called?
        mask = np.squeeze(el[1])
        img = np.moveaxis(el[0], 0, -1)
        patches = sample_patches([mask, img], (size, size), numPatches)
        for j, patch in enumerate(patches):
            resizedImgs[i*numPatches + j] = torch.from_numpy(patch[1])
            resizedLabels[i*numPatches + j] = torch.from_numpy(patch[0])

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
    # if numPatches >= 1:
    #     for i in range(resizedImgs.shape[0]):
    #         cv2.imshow('debug', resizedImgs[i, :].T.numpy().astype(np.float32))
    #         cv2.imshow('debug2', resizedLabels[i, :].T.numpy().astype(np.float32))
    #         cv2.waitKey(0)
    return resizedImgs, resizedLabels
