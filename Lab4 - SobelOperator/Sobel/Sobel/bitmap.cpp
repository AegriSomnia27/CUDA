#include "bitmap.h"

#include <iostream>
#include <fstream>

// Colour class
Colour::Colour() : red(0.0f), green(0.0f), blue(0.0f) {}
Colour::Colour(float r, float g, float b) : red(r), green(g), blue(b) {}
Colour::~Colour() {}


// Bitmap class
// constants for headers
const int Bitmap::BYTES_PER_PIXEL = 3;
const int Bitmap::FILE_HEADER_SIZE = 14;
const int Bitmap::INFO_HEADER_SIZE = 40;

// Constructor that creates a 2D array of pixels with their colours
Bitmap::Bitmap(int Height, int Width, const char* ImageName) :
	height(Height), width(Width), imageFileName(ImageName) {
	size = width * height * BYTES_PER_PIXEL;
	paddingAmount = ((4 - (width * 3) % 4) % 4);

	image = new Colour * [height];

	for (int i = 0; i < height; i++) {
		image[i] = new Colour[width];
	}
}

Bitmap::Bitmap(const char* inputFileName, const char* outputFileName): 
	height(0), width(0), imageFileName(outputFileName) {
	std::ifstream inputFile;
	inputFile.open(inputFileName, std::ios::binary);

	if (!inputFile.is_open()) {
		std::cerr << "File cannot be opened" << std::endl;
		return;
	}

	unsigned char fileHeader[FILE_HEADER_SIZE];
	inputFile.read(reinterpret_cast<char*>(fileHeader), FILE_HEADER_SIZE);

	if (fileHeader[0] != 'B' || fileHeader[1] != 'M') {
		std::cerr << "Unsupported type of a file!" << std::endl;
		inputFile.close();
		return;
	}

	unsigned char infoHeader[INFO_HEADER_SIZE];
	inputFile.read(reinterpret_cast<char*>(infoHeader), INFO_HEADER_SIZE);

	//size = fileHeader[2] + (fileHeader[3] << 8) + (fileHeader[4] << 16) + (fileHeader[5] << 24);
	width = infoHeader[4] + (infoHeader[5] << 8) + (infoHeader[6] << 16) + (infoHeader[7] << 24);
	height = infoHeader[8] + (infoHeader[9] << 8) + (infoHeader[10] << 16) + (infoHeader[11] << 24);
	size = width * height * BYTES_PER_PIXEL;

	paddingAmount = ((4 - (width * 3) % 4) % 4);

	image = new Colour * [height];
	for (int i = 0; i < height; i++) {
		image[i] = new Colour[width];
	}

	// Calculate offset to the pixels data
	const int offset = fileHeader[10] + (fileHeader[11] << 8) + (fileHeader[12] << 16) + (fileHeader[13] << 24);
	inputFile.seekg(offset, inputFile.beg);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned char* colour = new unsigned char[3];
			//unsigned char colour[3];
			inputFile.read(reinterpret_cast<char*>(colour), 3);
	
			image[y][x].red = static_cast<float>(colour[2]) / 255.0f;
			image[y][x].green = static_cast<float>(colour[1]) / 255.0f;
			image[y][x].blue = static_cast<float>(colour[0]) / 255.0f;

			delete[] colour;
		}
		inputFile.ignore(paddingAmount);
	}

	inputFile.close();

	std::cout << "File has been succesfully read!" << std::endl;
}

// Destructor that deletes an object and free the allocated memory
Bitmap::~Bitmap() {
	for (int i = 0; i < height; i++) {
		delete[] image[i];
	}
	delete[] image;
}

Colour Bitmap::GetColour(int x, int y) const {
	return image[y][x];
}

void Bitmap::SetColour(const Colour& colour, int x, int y) {
	image[y][x].red = colour.red;
	image[y][x].green = colour.green;
	image[y][x].blue = colour.blue;
}

int Bitmap::GetImageHeight() const{
	return height;
}

int Bitmap::GetImageWidth() const{
	return width;
}

int Bitmap::GetImageSize() const{
	return size;
}

float* Bitmap::GenerateLinearizedPixelMap() const{
	// If we want linearized array e.g. for GPU computation
	// pixelMap[i] = red channel; pixelMap[i+1] = green channel; pixelMap[i+2] = blue channel.

	float* pixelMap = new float[height * width * BYTES_PER_PIXEL];

	/*for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++){
			pixelMap[width * y + x] = this->GetColour(x, y).red;
			pixelMap[width*height + width * y + x] = this->GetColour(x, y).green;
			pixelMap[2*width*height +width * y + x] = this->GetColour(x, y).blue;
		}
	}*/

	//x + WIDTH * (y + DEPTH * z)
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int z = 0; z < BYTES_PER_PIXEL; z++) {
				if (z == 0) {
					pixelMap[y * width * BYTES_PER_PIXEL + x * BYTES_PER_PIXEL + z] = this->GetColour(x, y).red;
				} else if (z == 1) {
					pixelMap[y * width * BYTES_PER_PIXEL + x * BYTES_PER_PIXEL + z] = this->GetColour(x, y).green;
				} else {
					pixelMap[y * width * BYTES_PER_PIXEL + x * BYTES_PER_PIXEL + z] = this->GetColour(x, y).blue;
				}
			}
		}
	}

	return pixelMap;
}

float** Bitmap::Generate2DPixelMap() const {
	// If we want 2D array. MAYBE in some cases we might need that
	// pixelMap[y][i] = red channel; pixelMap[y][i+1] = green channel; pixelMap[y][i+2] = blue channel.

		float** pixelMap = new float* [height];

		for (int y = 0; y < height; y++) {
			pixelMap[y] = new float[width * BYTES_PER_PIXEL];

			for (int x = 0, i = 0; x < width; i += 3, x++) {
				pixelMap[y][i] = this->GetColour(x, y).red;
				pixelMap[y][i + 1] = this->GetColour(x, y).green;
				pixelMap[y][i + 2] = this->GetColour(x, y).blue;
			}
		}

		return pixelMap;
}

void Bitmap::InitializeWithLinearizedPixelMap(float* pixelMap, const int pixelMapHeight, const int pixelMapWidth){
	// Check if a current object has different height and width
	if (height != pixelMapHeight || width != pixelMapWidth / 3) {
		std::cout << "The current bmp object contains a different image\n";
		std::cout << "deleting the previous image...\n";
		// free allocated memory by **image
		this->~Bitmap();

		// reloade attributes
		std::cout << "creating new image...\n";

		new(this) Bitmap(pixelMapHeight, pixelMapWidth / BYTES_PER_PIXEL);
	}
	
	for (int y = 0; y < pixelMapHeight; y++) {
		for (int x = 0; x < width; x++) {
			Colour currentColour;
			for (int z = 0; z < BYTES_PER_PIXEL; z++) {
				if (z == 0) {
					currentColour.red = pixelMap[y*width*BYTES_PER_PIXEL + x*BYTES_PER_PIXEL + z];
				}else if (z == 1) {
					currentColour.green = pixelMap[y * width * BYTES_PER_PIXEL + x * BYTES_PER_PIXEL + z];
				}else {
					currentColour.blue = pixelMap[y * width * BYTES_PER_PIXEL + x * BYTES_PER_PIXEL + z];
				}
			}
			this->SetColour(currentColour, x, y);
		}
	}

}

void Bitmap::InitializeWith2DPixelMap(float** pixelMap, const int pixelMapHeight, const int pixelMapWidth){
	// Check if a current object has different height and width
	if (height != pixelMapHeight || width != pixelMapWidth/3) {
		std::cout << "The current bmp object contains a different image\n";
		std::cout << "deleting the previous image...\n";
		// free allocated memory by **image
		this->~Bitmap();
		
		// reloade attributes
		std::cout << "creating new image...\n";

		new(this) Bitmap(pixelMapHeight, pixelMapWidth / 3);
	}
	

	for (int y = 0; y < pixelMapHeight; y++) {
		for (int x = 0, i = 0; x < width; i+=3, x++) {
			this->SetColour(
				Colour(pixelMap[y][i], pixelMap[y][i + 1], pixelMap[y][i + 2]),
				x,
				y
			);
		}
	}

	// Free memory allocated by pixelMap
	// TODO: needs a better implementation, but for now this will do
	// this->FreePixelMapMemory(pixelMap, pixelMapHeight, pixelMapWidth);
}

void Bitmap::GenerateBitmapImage(const char* fileName){
	imageFileName = fileName;

	unsigned char bmpPad[3] = { 0, 0, 0 };

	std::ofstream file;
	file.open(imageFileName, std::ios::out | std::ios::binary);

	if (!file.is_open()) {
		std::cerr << "File cannot be opened";
		return;
	}

	unsigned char* fileHeader = CreateBitmapFileHeader();
	unsigned char* informationHeader = CreateBitmapInfoHeader();

	file.write(reinterpret_cast<char*>(fileHeader), FILE_HEADER_SIZE);
	delete[] fileHeader;

	file.write(reinterpret_cast<char*>(informationHeader), INFO_HEADER_SIZE);
	delete[] informationHeader;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned char r = static_cast<unsigned char>(GetColour(x, y).red * 255.0f);		// Red channel is a floating point number
			unsigned char g = static_cast<unsigned char>(GetColour(x, y).green * 255.0f);	// that's the reason why we multiply it by 255.0f,
			unsigned char b = static_cast<unsigned char>(GetColour(x, y).blue * 255.0f);	// so we can get into the interval (0; 255)

			unsigned char pixelColour[] = { b, g, r }; // write pixels start with a blue channel

			file.write(reinterpret_cast<char*>(pixelColour), 3);
		}

		file.write(reinterpret_cast<char*>(bmpPad), paddingAmount);
	}

	file.close();

	std::cout << "File has been created!\n";
}

void Bitmap::DisplayImageInfo() const{
	std::cout << "Height: " << height << std::endl;
	std::cout << "Width: " << width << std::endl;
	std::cout << "Size: " << size << std::endl;
	std::cout << "Padding: " << paddingAmount << std::endl;
}

unsigned char* Bitmap::CreateBitmapFileHeader()const {
	const int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + width * height * BYTES_PER_PIXEL + paddingAmount * width;

	// unsigned char fileHeader[FILE_HEADER_SIZE];
	unsigned char* fileHeader = new unsigned char[FILE_HEADER_SIZE];

	// File type, must always be set to 'BM' to declare that this is a .bmp-file
	fileHeader[0] = 'B';
	fileHeader[1] = 'M';
	// File size in bytes
	fileHeader[2] = static_cast<unsigned char>(fileSize);
	fileHeader[3] = static_cast<unsigned char>(fileSize >> 8);
	fileHeader[4] = static_cast<unsigned char>(fileSize >> 16);
	fileHeader[5] = static_cast<unsigned char>(fileSize >> 24);
	// Reserved 1, must always be set to zero
	fileHeader[6] = static_cast<unsigned char>(0);
	fileHeader[7] = static_cast<unsigned char>(0);
	// Reserved 2, must always be set to zero
	fileHeader[8] = static_cast<unsigned char>(0);
	fileHeader[9] = static_cast<unsigned char>(0);
	// Pixel data offset from the beginning of the file to the bitmap data
	fileHeader[10] = static_cast<unsigned char>(FILE_HEADER_SIZE + INFO_HEADER_SIZE);
	fileHeader[11] = static_cast<unsigned char>(0);
	fileHeader[12] = static_cast<unsigned char>(0);
	fileHeader[13] = static_cast<unsigned char>(0);

	return fileHeader;
}

unsigned char* Bitmap::CreateBitmapInfoHeader() const {
	unsigned char* informationHeader = new unsigned char[INFO_HEADER_SIZE];

	// Header size
	informationHeader[0] = static_cast<unsigned char>(INFO_HEADER_SIZE);
	informationHeader[1] = static_cast<unsigned char>(0);
	informationHeader[2] = static_cast<unsigned char>(0);
	informationHeader[3] = static_cast<unsigned char>(0);
	// Image width, in pixels
	informationHeader[4] = static_cast<unsigned char>(width);
	informationHeader[5] = static_cast<unsigned char>(width >> 8);
	informationHeader[6] = static_cast<unsigned char>(width >> 16);
	informationHeader[7] = static_cast<unsigned char>(width >> 24);
	// Image height, in pixels
	informationHeader[8] = static_cast<unsigned char>(height);
	informationHeader[9] = static_cast<unsigned char>(height >> 8);
	informationHeader[10] = static_cast<unsigned char>(height >> 16);
	informationHeader[11] = static_cast<unsigned char>(height >> 24);
	// Planes of the target device, must be set to zero
	informationHeader[12] = static_cast<unsigned char>(1);
	informationHeader[13] = static_cast<unsigned char>(0);
	// Bits per pixel (RGB)
	informationHeader[14] = static_cast<unsigned char>(24);
	informationHeader[15] = static_cast<unsigned char>(0);
	// Compression, usually set to zero (No compression)
	informationHeader[16] = static_cast<unsigned char>(0);
	informationHeader[17] = static_cast<unsigned char>(0);
	informationHeader[18] = static_cast<unsigned char>(0);
	informationHeader[19] = static_cast<unsigned char>(0);
	// Image size, in bytes (No compression)
	informationHeader[20] = static_cast<unsigned char>(0);
	informationHeader[21] = static_cast<unsigned char>(0);
	informationHeader[22] = static_cast<unsigned char>(0);
	informationHeader[23] = static_cast<unsigned char>(0);
	// Horizontal pixels per meter (Not specified)
	informationHeader[24] = static_cast<unsigned char>(0);
	informationHeader[25] = static_cast<unsigned char>(0);
	informationHeader[26] = static_cast<unsigned char>(0);
	informationHeader[27] = static_cast<unsigned char>(0);
	// Vertical pixels per meter (Not specified)
	informationHeader[28] = static_cast<unsigned char>(0);
	informationHeader[29] = static_cast<unsigned char>(0);
	informationHeader[30] = static_cast<unsigned char>(0);
	informationHeader[31] = static_cast<unsigned char>(0);
	// Total colours (Colour palette not used)
	informationHeader[32] = static_cast<unsigned char>(0);
	informationHeader[33] = static_cast<unsigned char>(0);
	informationHeader[34] = static_cast<unsigned char>(0);
	informationHeader[35] = static_cast<unsigned char>(0);
	// Important colours (Generally ignored)
	informationHeader[36] = static_cast<unsigned char>(0);
	informationHeader[37] = static_cast<unsigned char>(0);
	informationHeader[38] = static_cast<unsigned char>(0);
	informationHeader[39] = static_cast<unsigned char>(0);

	return informationHeader;
}

void Bitmap::ClearImageData(){

}

void Bitmap::FreePixelMapMemory(float** pixelMap, int pixelMapHeight, int pixelMapWidth){
	for (int y = 0; y < pixelMapHeight; y++) {
		delete[] pixelMap[y];
	}
	delete[] pixelMap;
}