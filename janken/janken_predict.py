import tensorflow as tf
import numpy as np
from janken_train import target_size
from janken_train import batch_size
from janken_train import preprocessing_function

import tensorflow_hub as hub
import matplotlib.pyplot as plt


def plot(pred_confidence, pred_class, test_gen):
    # 最初のバッチを取得
    images, _ = next(test_gen)

    # 最初の画像を取得
    first_image = images[0]
    first_image = first_image.astype('float32') / 255

    # 画像を表示
    plt.imshow(first_image)
    plt.title(f'Predicted class: {pred_class[0]}, Confidence: {pred_confidence[0][pred_class[0]]}')
    plt.show()
    
def main():
    # 評価用ImageDataGenerator作成
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # preprocessing_function=preprocessing_function
    )
    # 評価用ジェネレータ作成
    test_gen = test_datagen.flow_from_directory(
        "images",
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # 学習済みモデルロード
    model = tf.keras.models.load_model("test_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})

    # 予測実施
    pred_confidence = model.predict(test_gen)
    pred_class = np.argmax(pred_confidence, axis=1)
    np.array(pred_confidence)

    # 予測結果ファイル出力
    print(pred_class)
    np.savetxt("result.csv", pred_class, fmt="%d")
    plot(pred_confidence, pred_class, test_gen)


if __name__ == "__main__":
    main()
