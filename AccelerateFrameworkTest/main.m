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
            row = [row stringByAppendingFormat:@"%.10f ", *(matrix + i*columns + j)];
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

double one = 1, zero = 0;

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSString *path = [[NSProcessInfo processInfo] environment][@"path"];

        NSInputStream *trainingImagesInputStream = [NSInputStream inputStreamWithFileAtPath:[path stringByAppendingPathComponent:@"train-images.idx3-ubyte"]];
        [trainingImagesInputStream open];

        uint8_t *buffer32 = malloc(4 * sizeof(uint8_t));

        unsigned long images, rows, columns, pixels;

        [trainingImagesInputStream read:buffer32 maxLength:4]; //skip msb

        [trainingImagesInputStream read:buffer32 maxLength:4];
        images = unpack_uint32(buffer32);

        [trainingImagesInputStream read:buffer32 maxLength:4];
        rows = unpack_uint32(buffer32);

        [trainingImagesInputStream read:buffer32 maxLength:4];
        columns = unpack_uint32(buffer32);

        pixels = rows * columns;

        images = 1;

        double *data = malloc(images * pixels * sizeof(double));

        uint8_t buffer8;

        unsigned long image_cnt = 0;
        int pixel_cnt = 0;

        while ([trainingImagesInputStream hasBytesAvailable]) {
            [trainingImagesInputStream read:&buffer8 maxLength:1];

            if (pixel_cnt > pixels - 1) {
                pixel_cnt = 0;
                image_cnt++;

                if (image_cnt == images) {
                    break;
                }
            }

            *(data + image_cnt * pixels + pixel_cnt++) = buffer8;
        }

        NSInputStream *trainingLabelsInputStream = [NSInputStream inputStreamWithFileAtPath:[path stringByAppendingPathComponent:@"train-labels.idx1-ubyte"]];
        [trainingLabelsInputStream open];

        double *labels = malloc(images * 10 * sizeof(double));

        image_cnt = 0;

        while ([trainingLabelsInputStream hasBytesAvailable] && image_cnt < images) {
            [trainingImagesInputStream read:&buffer8 maxLength:1];

            vDSP_vclrD(&labels[image_cnt * 10], 1, 10);
            *(labels + image_cnt * 10 + (unsigned long)buffer8) = 1;
            image_cnt++;
        }

        unsigned long l_a1 = pixels, l_a2 = 10, l_a3 = 10;
        double *X = malloc(images * (l_a1 + 1) * sizeof(double));

        vDSP_mmovD(data, &X[1], l_a1 + 1, images, l_a1, l_a1 + 1);
        vDSP_vrampD(&one, &zero, X, pixels + 1, images);

        printf("X:\n");
        printDoubleMatrix(X, images, pixels + 1);
        printf("\n\n");

        double *Theta1 __attribute__ ((aligned)) = malloc((l_a1 + 1) * l_a2 * sizeof(double));
        double *Theta2 __attribute__ ((aligned)) = malloc((l_a1 + 1) * l_a2 * sizeof(double));

        randomizeWeights(Theta1, l_a1 + 1, l_a2);
        randomizeWeights(Theta2, l_a2 + 1, l_a3);

        printf("Theta1 matrix:\n");
        printDoubleMatrix(Theta1, l_a1 + 1, l_a2);
        printf("\n\n");

        printf("Theta2 matrix:\n");
        printDoubleMatrix(Theta2, l_a2 + 1, l_a3);
        printf("\n\n");

        double *z2 __attribute__ ((aligned)) = malloc(images * l_a2 * sizeof(double));
        double *a2 __attribute__ ((aligned)) = malloc(images * (l_a2 + 1) * sizeof(double));
        double *z3 __attribute__ ((aligned)) = malloc(images * l_a3 * sizeof(double));
        double *a3 __attribute__ ((aligned)) = malloc(images * l_a3 * sizeof(double));

        double *Theta1T = malloc((l_a1 + 1) * l_a2 * sizeof(double));
        vDSP_mtransD(Theta1, 1, Theta1T, 1, l_a1 + 1, l_a2);

        vDSP_mmulD(X, 1, Theta1, 1, z2, 1, images, l_a2, (l_a1 + 1));

        double sum = 0;
        for (unsigned long i = 0; i < pixels + 1; i++) {
            sum += (*(X + i) * *(Theta1 + 10*i));
        }

        printf("z2[0] = %f\n\n\n", sum);

#define MATRIX_DEBUG
#ifdef MATRIX_DEBUG
        printf("z2 matrix:\n");
        printDoubleMatrix(z2, images, (uint32_t) l_a2);
        printf("\n\n");
#endif

        unsigned long lm_z2 = l_a2 * images;

        vDSP_vnegD(z2, 1, z2, 1, lm_z2);
        vvexp(z2, z2, (int const *) &lm_z2);

#ifdef MATRIX_DEBUG
        printf("exp(-z2)\n");
        printDoubleMatrix(z2, images, l_a2);
        printf("\n\n");
#endif

        vDSP_vsaddD(z2, 1, &one, z2, 1, lm_z2);

#ifdef MATRIX_DEBUG
        printf("1 + exp(-z2)\n");
        printDoubleMatrix(z2, images, l_a2);
        printf("\n\n");
#endif

        double *ones = malloc(lm_z2 * sizeof(double));
        vDSP_vrampD(&one, &zero, ones, 1, lm_z2);

        vDSP_vdivD(z2, 1, ones, 1, z2, 1, lm_z2);

#ifdef MATRIX_DEBUG
        printf("1 / (1 + exp(-z2))\n");
        printDoubleMatrix(z2, images, l_a2);
        printf("\n\n");
#endif

        vDSP_mmovD(z2, &a2[1], l_a2 + 1, images, l_a2, l_a2 + 1);
        vDSP_vrampD(&one, &zero, a2, l_a2 + 1, images);

#ifdef MATRIX_DEBUG
        printf("a2 with bias unit:\n");
        printDoubleMatrix(a2, images, l_a2 + 1);
        printf("\n\n");
#endif

        double *Theta2T = malloc((l_a1 + 1) * l_a2 * sizeof(double));
        vDSP_mtransD(Theta2, 1, Theta2T, 1, l_a2 + 1, l_a3);

        vDSP_mmulD(a2, 1, Theta2, 1, z3, 1, images, l_a3, (l_a2 + 1));

#ifdef MATRIX_DEBUG
        printf("z3:\n");
        printDoubleMatrix(z3, images, l_a3);
        printf("\n\n");
#endif

        unsigned long lm_z3 = l_a3 * images;

        vDSP_vnegD(z3, 1, z3, 1, lm_z3);
        vvexp(z3, z3, (int const *) &lm_z3);

#ifdef MATRIX_DEBUG
        printf("exp(-z3)\n");
        printDoubleMatrix(z3, images, l_a2);
        printf("\n\n");
#endif

        vDSP_vsaddD(z3, 1, &one, z3, 1, lm_z3);

#ifdef MATRIX_DEBUG
        printf("1 + exp(-z3)\n");
        printDoubleMatrix(z3, images, l_a2);
        printf("\n\n");
#endif

        vDSP_vdivD(z3, 1, ones, 1, z3, 1, lm_z3);

#ifdef MATRIX_DEBUG
        printf("1 / (1 + exp(-z3))\n");
        printDoubleMatrix(z3, images, l_a3);
        printf("\n\n");
#endif
    }
    return 0;
}

