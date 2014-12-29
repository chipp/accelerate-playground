//
//  main.m
//  AccelerateFrameworkTest
//
//  Created by Vladimir Burdukov on 12/24/14.
//
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

void printIntMatrix(uint8_t **matrix, uint32_t rows, uint32_t columns) {
    for (uint32_t i = 0; i < rows; i++) {
        NSString *row = @"";
        for (uint32_t j = 0; j < columns; j++) {
            uint8_t value = matrix[i][j];
            row = [row stringByAppendingFormat:@"%d\t", value];
        }

        NSLog(@"%@\n", row);
    }
}

void printDoubleMatrix(double **matrix, uint8_t rows, uint8_t columns) {
    for (uint8_t i = 0; i < rows; i++) {
        NSString *row = @"";
        for (uint8_t j = 0; j < columns; j++) {
            double value = matrix[i][j];
            row = [row stringByAppendingFormat:@"%.3f\t", value];
        }

        NSLog(@"%@\n", row);
    }
}

uint32_t unpack_uint32(uint8_t values[4]) {
    return 0x1000000 * (uint32_t) values[0] + 0x10000 * (uint32_t) values[1] + 0x100 * (uint32_t) values[2] + (uint32_t) values[3];
}

void randomizeWeights(double **weights, int l_in, int l_out) {
    double eps_init = sqrt(6) / sqrt(l_in + l_out);

    for (int i = 0; i < l_in; i++) {
        for (int j = 0; j < l_out; j++) {
            weights[i][j] = ((double)arc4random() / 0x100000000) * 2 * eps_init - eps_init;
        }
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
//        float a[8][4] =
//                {
//                        {0.341581, 0.625796, 0.904949, 0.852354},
//                        {0.636449, 0.161477, 0.615051, 0.279334},
//                        {0.029207, 0.691676, 0.573400, 0.813822},
//                        {0.970948, 0.299834, 0.316818, 0.570824},
//                        {0.432712, 0.093691, 0.528665, 0.809598},
//                        {0.400032, 0.115675, 0.560931, 0.792352},
//                        {0.226343, 0.525194, 0.429942, 0.534593},
//                        {0.883902, 0.034078, 0.457998, 0.649731}
//                };
//
//        float x[4] = {0.15833, 0.33001, 0.78909, 0.80225};
//        float y[8];
//
//        vDSP_mmul(a, 1, x, 1, &y, 1, 8, 1, 4);

        NSInputStream *trainingImagesInputStream = [NSInputStream inputStreamWithFileAtPath:@"/Users/chipp/Developer/accelerate-playground/train-images.idx3-ubyte"];
        [trainingImagesInputStream open];

        uint8_t *buffer32 = malloc(4 * sizeof(uint8_t));

        uint32_t imagesCount, rowsCount, columnsCount;

        [trainingImagesInputStream read:buffer32 maxLength:4]; //skip msb

        [trainingImagesInputStream read:buffer32 maxLength:4];
        imagesCount = unpack_uint32(buffer32);

        [trainingImagesInputStream read:buffer32 maxLength:4];
        rowsCount = unpack_uint32(buffer32);

        [trainingImagesInputStream read:buffer32 maxLength:4];
        columnsCount = unpack_uint32(buffer32);

        imagesCount = 5;

        uint8_t **data = (uint8_t **)malloc(imagesCount * sizeof(uint8_t *));
        for (int i = 0; i < imagesCount; i++) {
            data[i] = (uint8_t *)malloc(rowsCount * columnsCount * sizeof(uint8_t));
        }

        uint8_t buffer8;

        uint32_t imageCounter = 0;
        int pixelCounter = 0;

        while ([trainingImagesInputStream hasBytesAvailable]) {
            [trainingImagesInputStream read:&buffer8 maxLength:1];

            if (pixelCounter > (rowsCount * columnsCount - 1)) {
                pixelCounter = 0;
                imageCounter++;

                if (imageCounter == imagesCount) {
                    break;
                }
            }

            data[imageCounter][pixelCounter++] = buffer8;
        }

        NSInputStream *trainLabelsInputStream = [NSInputStream inputStreamWithFileAtPath:@"/Users/chipp/Developer/accelerate-playground/train-labels.idx1-ubyte"];
        [trainLabelsInputStream open];

        uint8_t *labels = (uint8_t *)malloc(imagesCount * sizeof(uint8_t));

        imageCounter = 0;

        while ([trainingImagesInputStream hasBytesAvailable] && imageCounter < imagesCount) {
            [trainingImagesInputStream read:&buffer8 maxLength:1];

            labels[imageCounter++] = buffer8;
        }

        int l_in = 784, l_out = 25;

        double **weights = (double **)malloc(l_in * sizeof(double *));
        for (int i = 0; i < l_in; i++) {
            weights[i] = (double *)malloc(l_out * sizeof(double));
        }

        randomizeWeights(weights, l_in, l_out);

        double *a2 = (double *)malloc(l_out * sizeof(double));

        vDSP_mmulD((double const *) weights, 1, (double const *) data[1], 1, a2, 1, (vDSP_Length) l_in, 1, (vDSP_Length) l_out);

        for (int i = 0; i < l_out; i++) {
            NSLog(@"%.20f", a2[i]);
        }
    }
    return 0;
}

