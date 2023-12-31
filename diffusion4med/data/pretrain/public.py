# do actually need dependecies from vox2vec?
from vox2vec.pretrain.data import Public
from imops import crop_to_box
from vox2vec.processing import scale_hu, sample_box


class ModifiedPublic(Public):
    def __getitem__(self, i):
        image, body_mask = self.load_example(self.ids[i])

        box = sample_box(image.shape, self.patch_size)
        image, body_mask = crop_to_box(image, box), crop_to_box(body_mask, box)
        # image_x, image_y, image_z = image.shape
        # image = image[
        #     image_x // 2
        #     - self.patch_size[0] // 2 : image_x // 2
        #     + self.patch_size[0] // 2,
        #     image_y // 2
        #     - self.patch_size[1] // 2 : image_y // 2
        #     + self.patch_size[1] // 2,
        #     image_z // 2
        #     - self.patch_size[2] // 2 : image_z // 2
        #     + self.patch_size[2] // 2,
        # ]

        image = scale_hu(image, self.window_hu)
        image = image * 2 - 1

        image = image[None]  # add channel dim

        return image
