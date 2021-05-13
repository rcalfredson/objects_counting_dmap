if glob_models:
        model_paths = glob.glob(model_path)
        print('model paths:', model_paths)
        predictions_by_model = []
        for i, model_path in enumerate(model_paths):
            predictions_by_model.append([])
            network = {
                'UNet': UNet,
                'FCRN_A': FCRN_A,
                'FCRN_B': FCRN_B
            }[network_architecture](input_filters=input_channels,
                                    filters=unet_filters,
                                    N=convolutions,
                                    dropout=dropout).to(device)
            network = torch.nn.DataParallel(network)
            # network.load_state_dict(torch.load(model_path))
            network.load_state_dict(torch.load(model_path))
            print('loaded net:', network)
            data_path = os.path.join(dataset_name, 'valid.h5')
            dataset = MultiDimH5Dataset(data_path)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            network.train(False)
            counter = 0
            for image, label in dataloader:
                image = image.to(device)
                label = label.to(device)
                result = network(image)
                for true, predicted in zip(label, result):
                    true_counts = torch.sum(true).item() / 100
                    predicted_counts = torch.sum(predicted).item() / 100
                    print('what is the sum of the image\'s right border?')
                    print(torch.sum(predicted[:, -3:])/ 100)
                    print('what is the sum of the inner 3/4 along both boundaries?')
                    oneEighthImgHt = int(predicted.shape[1] / 8)
                    oneEighthImgWidth = int(predicted.shape[2] / 8)
                    innerThreeFourths = predicted[:,
                        oneEighthImgHt:predicted.shape[1] - oneEighthImgHt,
                        oneEighthImgWidth:predicted.shape[2] - oneEighthImgWidth]
                    print(torch.sum(innerThreeFourths)/ 100)
                    print('true counts:', true_counts)
                    print('predicted counts:', predicted_counts)
                    if i == 0:
                        true_values.append(true_counts)
                    predictions_by_model[-1].append(predicted_counts)
                    if abs(true_counts - predicted_counts) > 6:
                        pass
                        exampleId = randID()
                        imgAsNp = image.cpu().numpy()
                        print('imgAsNp', imgAsNp)
                        dMapToShow = predicted.cpu().detach().numpy()[0].T
                        # print('density map?', dMapToShow.shape)
                        cv2.imshow('image', cv2.cvtColor(imgAsNp[0].T, cv2.COLOR_BGR2RGB))
                        cv2.imshow('densityMap', dMapToShow)
                        cv2.waitKey(0)
                        # cv2.imwrite(os.path.join('error_examples',
                        #     '%s_%i_pred_%i_actual_img.png'%(exampleId, predicted_counts, true_counts)),
                        #     255*cv2.cvtColor(imgAsNp[0].T, cv2.COLOR_BGR2RGB))
                        # cv2.imwrite(os.path.join('error_examples',
                        #     '%s_%i_pred_%i_actual_map.png'%(exampleId, predicted_counts, true_counts)),
                        #     255*dMapToShow)
                counter += 1
        true_values = np.array(true_values)
        predicted_values_orig = np.mean(predictions_by_model, axis=0)
        predictions_by_model = np.array(predictions_by_model)
        predicted_values, confInts = zip(*[meanConfInt(predictions_by_model[:, i], asDelta=True)[:2] for i in \
            range(predictions_by_model.shape[1])])
        print('predicted values using original approach:', predicted_values_orig)
        predicted_values = np.array(predicted_values)
        print('predicted using new approach:', predicted_values)
        print('confidence ints:', confInts)
        # question: how to go from the confidence intervals to the original images?
        # the dataloader has already been traversed.
        confInts = np.array(confInts)
        smallestDeltas = np.argpartition(confInts, 5)[:5]
        largestDeltas = np.argpartition(confInts, -5)[-5:]
        print('true values:', true_values)
        print('predicted values:', predicted_values)
        abs_diff = np.abs(np.subtract(predicted_values, true_values))
        abs_rel_errors = np.divide(abs_diff, true_values)
        print('absolute error:', abs_diff)
        print('relative errors:', abs_rel_errors)
        print('mean absolute error:', np.mean(abs_diff))
        print('mean relative error:', np.mean(
            abs_rel_errors[abs_rel_errors != np.infty]))
        print('mean relative error, 0-10 eggs:',
            np.mean(abs_rel_errors[(abs_rel_errors != np.infty) & (true_values < 11)]))
        print('mean relative error, 11-40 eggs:', np.mean(abs_rel_errors[(
            abs_rel_errors != np.infty) & ((true_values >= 11) & (true_values < 41))]))
        print('mean relative error, 41+ eggs:',
            np.mean(abs_rel_errors[(abs_rel_errors != np.infty) & (true_values >= 41)]))
        print('maximum error:', max(abs_diff))
        print('indices of smallest confInt deltas:', smallestDeltas)
        print('and the actual values:', confInts[smallestDeltas])
        print('now largest delta indices:', largestDeltas)
        print('and their values:', confInts[largestDeltas])
        counter = 0
        for image, label in dataloader:
            # counter tells you which image
            # the actual confInt is just confInts[counter]
            isSmallest = counter in smallestDeltas
            isLargest = counter in largestDeltas
            if isSmallest or isLargest:
                print('found a match. Counter is', counter)
                print('isSmallest and isLargest', isSmallest, isLargest)
                # how to know what the value is?
                cv2.imwrite('error_examples/%s_confInt_%.6f.jpg'%('smallest' if\
                    isSmallest else 'largest', confInts[counter]), 255*cv2.cvtColor(image.cpu().numpy()[0].T, cv2.COLOR_BGR2RGB))
            counter += 1