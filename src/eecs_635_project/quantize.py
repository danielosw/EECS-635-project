"""
    python src/eecs_635_project/quantize.py
    python src/eecs_635_project/quantize.py --model models/asl_model_boosted.keras --num-calib 300

Outputs (in models/ by default):
    asl_int8.tflite   the quantized model
    asl_int8_model.cc the same model as a C array (for firmware)
    labels.txt        class names, one per line, in output-index order
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# quantize.py lives in src/eecs_635_project/, so the project root is two levels up
ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description='Quantize the ASL model to int8 TFLite.')
    p.add_argument('--model', type=Path, default=ROOT / 'models' / 'asl_final_boosted_model.keras')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'ASL_Gestures_36_Classes')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'models')
    p.add_argument('--num-calib', type=int, default=200,
                   help='Number of training images used to calibrate int8 ranges')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def make_representative_dataset(train_dir, size, n, seed):
    """Yields real training images, preprocessed exactly like training (rescale 1/255,
    no augmentation). The converter uses them to pick int8 ranges for every layer."""
    gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        train_dir, target_size=size, batch_size=1,
        class_mode=None, shuffle=True, seed=seed)

    def representative_dataset():
        for _ in range(n):
            yield [next(gen).astype(np.float32)]

    return representative_dataset


def write_c_array(data, path, name='g_model'):
    with open(path, 'w') as f:
        f.write('#include <cstdint>\n\n')
        f.write(f'alignas(16) const unsigned char {name}[] = {{\n')
        for i in range(0, len(data), 12):
            f.write('  ' + ', '.join(f'0x{b:02x}' for b in data[i:i + 12]) + ',\n')
        f.write('};\n')
        f.write(f'const unsigned int {name}_len = {len(data)};\n')


def tflite_accuracy(tflite_bytes, test_dir, size):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zero_point = inp['quantization']

    gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_dir, target_size=size, batch_size=1,
        class_mode='categorical', shuffle=False)

    correct = 0
    for i in range(gen.samples):
        x, y = gen[i]
        q = np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)
        interp.set_tensor(inp['index'], q)
        interp.invoke()
        pred = interp.get_tensor(out['index'])[0]
        correct += int(np.argmax(pred) == np.argmax(y[0]))
    return correct / gen.samples


def float_accuracy(model, test_dir, size):
    gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_dir, target_size=size, batch_size=32,
        class_mode='categorical', shuffle=False)
    return model.evaluate(gen, verbose=0)[1]


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    if not args.model.exists():
        raise SystemExit(f'Model not found: {args.model}\nTrain first with main.py, or pass --model.')
    train_dir = args.data_dir / 'train'
    test_dir = args.data_dir / 'test'
    for d in (train_dir, test_dir):
        if not d.exists():
            raise SystemExit(f'Dataset folder not found: {d}')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    h, w = model.input_shape[1:3]
    size = (h, w)
    print(f'Loaded {args.model.name}: input {size}, {model.count_params():,} parameters')

    # Export as a SavedModel, then convert (the most reliable path with Keras 3)
    saved_dir = args.output_dir / 'saved_model_tmp'
    model.export(str(saved_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(
        train_dir, size, args.num_calib, args.seed)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_path = args.output_dir / 'asl_int8.tflite'
    tflite_path.write_bytes(tflite_model)
    write_c_array(tflite_model, args.output_dir / 'asl_int8_model.cc')

    labels = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    (args.output_dir / 'labels.txt').write_text('\n'.join(labels) + '\n')

    # Details the firmware needs
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f'\nSaved {tflite_path}  ({len(tflite_model) / 1024:.0f} KB)')
    print(f'Input : shape={inp["shape"].tolist()} dtype={inp["dtype"].__name__} '
          f'scale={inp["quantization"][0]:.6f} zero_point={inp["quantization"][1]}')
    print(f'Output: shape={out["shape"].tolist()} dtype={out["dtype"].__name__} '
          f'scale={out["quantization"][0]:.6f} zero_point={out["quantization"][1]}')

    # Did quantization hurt accuracy?
    print('\nEvaluating on the test set (this takes a minute)...')
    f_acc = float_accuracy(model, test_dir, size)
    q_acc = tflite_accuracy(tflite_model, test_dir, size)
    print(f'Float model accuracy: {f_acc:.4f}')
    print(f'Int8  model accuracy: {q_acc:.4f}  (drop: {f_acc - q_acc:+.4f})')


if __name__ == '__main__':
    main()