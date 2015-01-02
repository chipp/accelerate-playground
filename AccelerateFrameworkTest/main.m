//
//  main.m
//  AccelerateFrameworkTest
//
//  Created by Vladimir Burdukov on 12/24/14.
//
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

//#define MATRIX_DEBUG

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

void printDoubleMatrix(double *matrix, unsigned long rows, unsigned long columns) {
    for (unsigned long i = 0; i < rows; i++) {
        NSString *row = @"";
        for (unsigned long j = 0; j < columns; j++) {
            row = [row stringByAppendingFormat:@"%.10f\t\t", *(matrix + i*columns + j)];
        }

        row = [row stringByAppendingString:@"\n"];

        printf([row cStringUsingEncoding:NSUTF8StringEncoding]);
    }
}


uint32_t unpack_uint32(uint8_t values[4]) {
    return 0x1000000 * (uint32_t) values[0] + 0x10000 * (uint32_t) values[1] + 0x100 * (uint32_t) values[2] + (uint32_t) values[3];
}

void randomizeWeights(double *weights, unsigned long l_in, unsigned long l_out) {
    double eps_init = sqrt(6) / sqrt(l_in + l_out);

    for (int i = 0; i < l_in; i++) {
        for (int j = 0; j < l_out; j++) {
            *(weights + i*l_out + j) = ((double)arc4random() / 0x100000000) * 2 * eps_init - eps_init;
//            *(weights + i*l_out + j) = fake_random[i*l_out + j];
        }
    }
}

void printDoubleVec(double *vec, unsigned long count) {
    NSString *out = @"";
    for (unsigned long i = 0; i < count; i++) {
        out = [out stringByAppendingFormat:@"%.3f\t", vec[i]];
    }

    NSLog(@"%@", out);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
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

        unsigned long trainingExamplesCount = 10;
        double one = 1, zero = 0;

        double *X = malloc(trainingExamplesCount * 61 * sizeof(double));

        double generatedX[600] = {
                0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1
        };

        vDSP_mmovD(generatedX, &X[1], 61, trainingExamplesCount, 60, 61);
        vDSP_vrampD(&one, &zero, X, 10, trainingExamplesCount);

//        printf("X:\n");
//        printDoubleMatrix(X, 10, 61);

        unsigned long l_a1 = 60, l_a2 = 25, l_a3 = 10;

        double *Theta1 __attribute__ ((aligned)) = malloc((l_a1 + 1) * l_a2 * sizeof(double));
        double *Theta2 __attribute__ ((aligned)) = malloc((l_a1 + 1) * l_a2 * sizeof(double));

        randomizeWeights(Theta1, l_a1 + 1, l_a2);
        randomizeWeights(Theta2, l_a2 + 1, l_a3);

        printf("Theta1 matrix:\n");
        printDoubleMatrix(Theta1, l_a1 + 1, l_a2);

        double *z2 __attribute__ ((aligned)) = malloc(trainingExamplesCount * l_a2 * sizeof(double));
        double *a2 __attribute__ ((aligned)) = malloc(trainingExamplesCount * (l_a2 + 1) * sizeof(double));
        double *z3 __attribute__ ((aligned)) = malloc(trainingExamplesCount * l_a3 * sizeof(double));
        double *a3 __attribute__ ((aligned)) = malloc(trainingExamplesCount * l_a3 * sizeof(double));

        double *Theta1T = malloc((l_a1 + 1) * l_a2 * sizeof(double));
        vDSP_mtransD(Theta1, 1, Theta1T, 1, l_a1 + 1, l_a2);

        vDSP_mmulD(X, 1, Theta1T, 1, z2, 1, trainingExamplesCount, l_a2, (l_a1 + 1));

#ifdef MATRIX_DEBUG
        NSLog(@"z2 matrix:");
        printDoubleMatrix(z2, trainingExamplesCount, (uint32_t) l_a2);
#endif

        unsigned long lm_z2 = l_a2 * trainingExamplesCount;

        vDSP_vnegD(z2, 1, z2, 1, lm_z2);
        vvexp(z2, z2, (int const *) &lm_z2);

#ifdef MATRIX_DEBUG
        NSLog(@"exp(-z2)");
        printDoubleMatrix(z2, trainingExamplesCount, l_a2);
#endif

        vDSP_vsaddD(z2, 1, &one, z2, 1, lm_z2);

#ifdef MATRIX_DEBUG
        NSLog(@"1 + exp(-z2)");
        printDoubleMatrix(z2, trainingExamplesCount, l_a2);
#endif

        double *ones = malloc(lm_z2 * sizeof(double));
        vDSP_vrampD(&one, &zero, ones, 1, lm_z2);

        vDSP_vdivD(z2, 1, ones, 1, z2, 1, lm_z2);

#ifdef MATRIX_DEBUG
        NSLog(@"1 / (1 + exp(-z2))");
        printDoubleMatrix(z2, trainingExamplesCount, l_a2);
#endif

        vDSP_mmovD(z2, &a2[1], l_a2 + 1, trainingExamplesCount, l_a2, l_a2 + 1);
        vDSP_vrampD(&one, &zero, a2, l_a2 + 1, trainingExamplesCount);

#ifdef MATRIX_DEBUG
        NSLog(@"a2 with bias unit");
        printDoubleMatrix(a2, trainingExamplesCount, l_a2 + 1);
#endif

        double *Theta2T = malloc((l_a1 + 1) * l_a2 * sizeof(double));
        vDSP_mtransD(Theta2, 1, Theta2T, 1, l_a2 + 1, l_a3);

        vDSP_mmulD(a2, 1, Theta2, 1, z3, 1, trainingExamplesCount, l_a3, (l_a2 + 1));
//#define MATRIX_DEBUG
//#ifdef MATRIX_DEBUG
//        NSLog(@"z3");
//        printDoubleMatrix(z3, trainingExamplesCount, l_a3);
//#endif

//        for (int i = 0; i < l_a2; i++) {
//            NSLog(@"%.3f", z2[i]);
//        }

//        double a1_[3] = {1, 48, 59};
//        double a2_[] = {
//                0.547, -0.042, 0.365, 0.032, -0.529, -0.234, -0.188, 0.060, 0.079, 0.476,
//                -0.670, -0.316, 0.121, -0.017, 0.536, -0.291, 0.478, 0.560, -0.379, 0.638,
//                0.255, 0.657, -0.653, 0.492, -0.211, -0.019, -0.275, -0.057, 0.080, -0.027
//        };
//
//        double *a3_ = malloc(10 * sizeof(double));
//
//        vDSP_mmulD(a1_, 1, (double const *) a2_, 1, a3_, 1, 1, 10, 3);
//
//        for (int i = 0; i < 10; i++) {
//            NSLog(@"%f", a3_[i]);
//        }
    }
    return 0;
}

