def vector_extraction_batch_full(self, data, batch_size):
    dataset = LoadImages(data, self.img_size, batch_size=batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    result_list = []

    for batch_i, (imgs, paths, shapes) in enumerate(tqdm(dataloader)):
        batch_time = time.time()
        torch.cuda.empty_cache()

        with torch.no_grad():
            imgs = imgs.to(self.device).float() / 255.0
            _, _, height, width = imgs.shape

            layerResult = LayerResult(self.model.module_list, 80)
            pred = self.model(imgs)[0]
            layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
            LayerResult.unregister_forward_hook(layerResult)

            for i in range(layerResult_tensor.shape[0]):
                MAC = max_pooling_tensor(layerResult_tensor[i])
                SPoC = average_pooling_tensor(layerResult_tensor[i])
                CAT = torch.cat((MAC, SPoC))

                data = {
                    'feature_vector_mac': MAC.detach().cpu().numpy(),
                    'feature_vector_spoc': SPoC.detach().cpu().numpy(),
                    'feature_vector_cat': CAT.detach().cpu().numpy(),
                    'img_path': paths[i],
                    'state': 'full'
                }

                result_list.append(data)

    batch_end = time.time() - batch_time
    print(" Inference time for a image : {}".format(batch_end / batch_size))
    print(" Inference time for batch image : {}".format(batch_end))

    return result_list