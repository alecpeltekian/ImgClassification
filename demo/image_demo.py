from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot, inference_nii_model, show_result_nii_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, img = inference_nii_model(model, args.img)
    print(result)

    show_result_nii_pyplot(img, result)


if __name__ == '__main__':
    main()
