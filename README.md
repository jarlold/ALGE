# A Light Gradient Engine - ALGE
This is a small autodifferentiation engine, which is meant to work on low memory devices. It should
be compatible with any device on the Arduino platform- provided it has adequate compute and memory.

## What's in it?
- Common arithmetatic operations such as addition, multiplication, logarithms, exponentiation, etc.
- Common vector, matrix, and tensor operations such as matrix multiplication, addition, im2col, etc.
- Convolutional and dense neural networks.
- A variety of optimizations techniques.
- Special numerical methods and compression techniques for increasing batch size on limited memory.

## What's it run on?
- Microprocessors such as the ESP32, AW-CU488, Arduino Uno, etc.
- Windows, Linux, and MacOS.
- Pretty much anything you can compile G++ for.

## What are the dependencies?
- Nothing I actually wrote the software myself (an advanced technique most developers cannot grasp).

## What's special about it?
It uses less program space than tensorflow lite and probably other autodifferentiation engines. I believe it uses less
memory too, but I didn't check everything else on the market.

## License
This work 
