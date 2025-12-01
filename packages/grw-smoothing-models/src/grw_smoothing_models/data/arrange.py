import os
from enum import Enum
from typing import List, Tuple
import csv
from more_itertools import peekable
import shutil


class FileType(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


class FileInfo:
    def __init__(self, name: str, id: str, class_name: str, start: int, end: int):
        self.name: str = name
        self.class_name: str = class_name
        self.id: str = id
        self.start: int = start
        self.end: int = end

    def __str__(self):
        return f'{self.name}, {self.class_name}, {self.id}, {self.start}, {self.end}'

    def __repr__(self):
        return f'{self.name}, {self.class_name}, {self.id}, {self.start}, {self.end}'


class KineticsDatasetAnnotationFile:
    def __init__(self, dataset_root_dir: str, file_type: FileType) -> None:
        self.dataset_root_dir: str = dataset_root_dir
        self.file_type: FileType = file_type
        prefix: str = KineticsDatasetAnnotationFile.__get_prefix(file_type)
        self.data_root: str = os.path.join(dataset_root_dir, prefix)
        self.annotations: List[List[FileInfo]] = (
            KineticsDatasetAnnotationFile._parse_file(
                os.path.join(dataset_root_dir, 'annotations', f'{prefix}.txt'), file_type))

    @staticmethod
    def _parse_file(file_path: str, file_type: FileType) -> List[List[FileInfo]]:
        with open(file_path, 'r') as file:
            it = peekable(file)
            res = []
            while it:
                class_name, metas = KineticsDatasetAnnotationFile.get_next_mp4_block(it, file_type)
                if class_name:
                    res.append(metas)
                else:
                    assert not it.peek()
        return res

    @staticmethod
    def get_next_mp4_block(it: peekable, file_type) -> Tuple[str, List[FileInfo]]:
        match file_type:
            case FileType.TRAIN:
                block_start_prefix = 'train/'
            case FileType.VAL:
                block_start_prefix = 'val/'
            case _:
                raise ValueError(f'{file_type} file type not supported')
        while it.peek() and not it.peek().startswith(block_start_prefix):
            next(it)
        if not it.peek():
            return '', []
        class_name = next(it).strip()[len(block_start_prefix):-1]
        metas = []
        while it and it.peek().strip():
            file_name = next(it).strip()
            id = file_name[:11]
            start = file_name[12:18]
            end = file_name[19:25]
            file_info = FileInfo(file_name, id, class_name, int(start), int(end))
            metas.append(file_info)
        return class_name, metas

    @staticmethod
    def __get_prefix(file_type: FileType):
        match file_type:
            case FileType.TRAIN:
                return 'train'
            case FileType.VAL:
                return 'val'

    def write_as_pytorchvideo_csv(self, file_name):
        with open(os.path.join(self.dataset_root_dir, 'annotations', file_name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(["file_name", "class_name"])  # Writing the headers
            for class_files_infos in self.annotations:
                for file_info in class_files_infos:
                    writer.writerow([os.path.join(self.data_root, file_info.name), file_info.class_name])


class KineticsDataSet:
    def __init__(self, root: str):
        self.root: str = root

    def arrange_as_pytorchvision_kinetics_dataset(self):
        kinetics_dataset_val_annotation_file = (
            KineticsDatasetAnnotationFile(self.root, FileType.VAL))
        kinetics_dataset_train_annotation_file = (
            KineticsDatasetAnnotationFile(self.root, FileType.TRAIN))
        if not os.path.isfile(os.path.join(self.root, 'annotations', 'val.csv')):
            print('creating val.csv')
            kinetics_dataset_val_annotation_file.write_as_pytorchvideo_csv('val.csv')
        if not os.path.isfile(os.path.join(self.root, 'annotations', 'train.csv')):
            print('creating train.csv')
            kinetics_dataset_train_annotation_file.write_as_pytorchvideo_csv('train.csv')
        for class_files in kinetics_dataset_val_annotation_file.annotations:
            file: FileInfo = class_files[0]
            class_name = file.class_name
            class_folder = os.path.join(self.root, 'val', class_name)
            os.makedirs(class_folder, exist_ok=True)
            for file in class_files:
                src = os.path.join(self.root, 'val', file.name)
                tgt = os.path.join(class_folder, file.name)
                if not os.path.isfile(src) and not os.path.isfile(tgt):
                    print(f'Warn: file {file.name} does not exists as {src} nor {tgt}')
                elif os.path.isfile(src) and os.path.isfile(tgt):
                    print(f'Warn: file {file.name} exists both as {src} and {tgt}')
                    os.remove(src)
                elif os.path.isfile(tgt):
                    print(f'File {file.name} exists as {tgt}')
                else:
                    shutil.move(os.path.join(self.root, 'val', file.name), os.path.join(class_folder, file.name))
        for class_files in kinetics_dataset_train_annotation_file.annotations:
            file: FileInfo = class_files[0]
            class_name = file.class_name
            class_folder = os.path.join(self.root, 'train', class_name)
            os.makedirs(class_folder, exist_ok=True)
            for file in class_files:
                src = os.path.join(self.root, 'train', file.name)
                tgt = os.path.join(class_folder, file.name)
                if not os.path.isfile(src) and not os.path.isfile(tgt):
                    print(f'Warn: file {file.name} does not exists as {src} nor {tgt}')
                elif os.path.isfile(src) and os.path.isfile(tgt):
                    print(f'Warn: file {file.name} exists both as {src} and {tgt}')
                    os.remove(src)
                elif os.path.isfile(tgt):
                    print(f'File {file.name} exists as {tgt}')
                else:
                    shutil.move(os.path.join(self.root, 'train', file.name), os.path.join(class_folder, file.name))


