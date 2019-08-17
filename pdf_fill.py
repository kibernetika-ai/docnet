import argparse
import json

import numpy as np
import pdfrw
import pdf2image
from PIL import Image


ANNOT_KEY = '/Annots'
ANNOT_FIELD_KEY = '/T'
ANNOT_VAL_KEY = '/V'
ANNOT_RECT_KEY = '/Rect'
SUBTYPE_KEY = '/Subtype'
WIDGET_SUBTYPE_KEY = '/Widget'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output', '-O', default='output.jpg')
    parser.add_argument(
        '--mode',
        default='get',
        choices=['get', 'set'],
    )
    parser.add_argument('--json', help='Path to file with values')

    return parser.parse_args()


def get_fields(template):
    fields = {}
    annotations = template.pages[0][ANNOT_KEY]
    for annotation in annotations:
        if annotation[SUBTYPE_KEY] == WIDGET_SUBTYPE_KEY:
            if annotation[ANNOT_FIELD_KEY]:
                key = annotation[ANNOT_FIELD_KEY][1:-1]
                fields[key] = annotation.V

    return fields


def set_fields(template: pdfrw.PdfReader, data: dict):
    annotations = template.pages[0][ANNOT_KEY]
    for annotation in annotations:
        if annotation[SUBTYPE_KEY] == WIDGET_SUBTYPE_KEY:
            if annotation[ANNOT_FIELD_KEY]:
                key = annotation[ANNOT_FIELD_KEY][1:-1]
                #print(key)
                if key in data.keys() and data[key] is not None:
                    print('Set {} to {}'.format(key,data[key]))
                    annotation.update(
                        pdfrw.PdfDict(V='{}'.format(data[key]))
                    )

    template.Root.AcroForm.update(pdfrw.PdfDict(NeedAppearances=pdfrw.PdfObject('true')))


def rotate(image, angle):
    im2 = image.convert('RGBA')
    # rotated image
    rot = im2.rotate(angle, expand=1)
    # a white image same size as rotated image
    fff = Image.new('RGBA', rot.size, (255,) * 4)
    # create a composite image using the alpha layer of rot as a mask
    out = Image.composite(rot, fff, rot)
    return out.convert('RGB')


def main():
    args = parse_args()

    template = pdfrw.PdfReader(args.input)
    if args.mode == 'get':
        fields = get_fields(template)
        print(json.dumps(fields, indent=2))
    elif args.mode == 'set':
        with open(args.json) as f:
            data = json.load(f)

        set_fields(template, data)

        pdfrw.PdfWriter().write(args.output, template)
        pages = pdf2image.convert_from_path(args.output, dpi=300)
        image = pages[0]
        # image = image.transform(
        #     (image.width, image.height),
        #     Image.QUAD,
        #     (50, 2, 0, image.height - 10, image.width - 100, image.height - 100, image.width,0)
        # )
        #image = rotate(image, -1.5)
        image.save(args.output, 'JPEG')

        print('Saved to {}'.format(args.output))


if __name__ == '__main__':
    main()