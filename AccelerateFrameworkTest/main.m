//
//  main.m
//  AccelerateFrameworkTest
//
//  Created by Vladimir Burdukov on 12/24/14.
//
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        float a[8][4] =     // the matrix to be multiplied
                {
                        {0.341581, 0.625796, 0.904949, 0.852354},
                        {0.636449, 0.161477, 0.615051, 0.279334},
                        {0.029207, 0.691676, 0.573400, 0.813822},
                        {0.970948, 0.299834, 0.316818, 0.570824},
                        {0.432712, 0.093691, 0.528665, 0.809598},
                        {0.400032, 0.115675, 0.560931, 0.792352},
                        {0.226343, 0.525194, 0.429942, 0.534593},
                        {0.883902, 0.034078, 0.457998, 0.649731}
                };

        float x[4] = {0.15833, 0.33001, 0.78909, 0.80225};  // the vector to be multiplied
        float y[8];

        vDSP_mmul(a, 1, x, 1, &y, 1, 8, 1, 4);

        for (int i = 0; i < 8; i++) {
            NSLog(@"%f", y[i]);
        }
    }
    return 0;
}
