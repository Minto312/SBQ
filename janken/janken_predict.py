import tensorflow as tf
import numpy as np
from janken_train import target_size
from janken_train import batch_size
from janken_train import preprocessing_function


def main():
    # 評価用ImageDataGenerator作成
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    # 評価用ジェネレータ作成
    test_gen = test_datagen.flow_from_directory(
        "img_test",
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # 学習済みモデルロード
    model = tf.keras.models.load_model("model.h5")

    # 予測実施
    pred_confidence = model.predict(test_gen)
    pred_class = np.argmax(pred_confidence, axis=1)

    # 予測結果ファイル出力
    print(pred_class)
    np.savetxt("result.csv", pred_class, fmt="%d")


if __name__ == "__main__":
    main()
