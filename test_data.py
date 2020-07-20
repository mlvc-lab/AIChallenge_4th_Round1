from pathlib import Path
from unittest import TestCase
from zipfile import ZipFile

import data 


class DataTest(TestCase):
    def __init__(self, *args, **kwargs):
        super(DataTest, self).__init__(*args, **kwargs)
        self.source = '/tmp/source'
        self.target = '/tmp/target'
        self.temp = '/tmp/things'

    def setUp(self):
        # create data dir and zip
        names = [
                    '상품_신발_운동화캐쥬얼화_운동화_패션운동화',
                    '상품_신발_남성화_운동화_패션운동화',
                    '상품_신발_여성화_운동화_패션운동화'
                ]

        # clean up data
        if Path(self.source).exists():
            self._rm_tree(self.source)

        # create dirs
        if not Path(self.source).exists():
            Path(self.source).mkdir()

        # make zip
        for num in range(3):
            zip_name = f'{self.source}/상품-{num}.zip'
            root = f"1-{num}"

            # clean up data
            if Path(zip_name).exists():
                Path(zip_name).unlink()
            if Path(root).exists():
                self._rm_tree(root)

            # zip file
            zip_file = ZipFile(zip_name, 'w')

            # make sample dir
            root = Path(root)
            root.mkdir()

            for sub in range(3):
                sub_dir = root / (f'HF02000{num}{sub}_' + names[num])
                sub_dir.mkdir()
                for image in range(50):
                    img = sub_dir / (f'HF02000{num}{sub}_' + str(image) + '.JPG')
                    img.touch()

                    # add to zip
                    zip_file.write(img)
            zip_file.close()
            self._rm_tree(root)

    def tearDown(self):
        # for path in [self.source, self.target, self.temp]:
        #     if Path(path).exists():
        #         self._rm_tree(path)
        pass

    def _rm_tree(self, pth):
        pth = Path(pth)
        for child in pth.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                self._rm_tree(child)
        pth.rmdir()

    def test_things_dataloader(self):
        self.test_zip()
        t_loader, v_loader = data.DataLoader(
            512,
            1,
            dataset="things",
            datapath=self.target,
            cuda=False
        )
        self.assertTrue(t_loader)
        self.assertTrue(v_loader)

        self.assertEqual(len(t_loader.dataset) + len(v_loader.dataset), 450)
        self.assertEqual(len(t_loader.dataset), 385)
        self.assertEqual(len(v_loader.dataset), 65)

    def test_zip(self):
        data.things_unzip_and_convert(self.source, self.target)
