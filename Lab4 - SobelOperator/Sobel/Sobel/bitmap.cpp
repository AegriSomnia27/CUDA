#include "bitmap.h"

#include <iostream>
#include <fstream>

// Colour class
Colour::Colour() : red(0.0f), green(0.0f), blue(0.0f) {}
Colour::Colour(float r, float g, float b) : red(r), green(g), blue(b) {}
Colour::~Colour() {}


// Bitmap class
//  онстанты дл€ записи метаданных
const int Bitmap::BYTES_PER_PIXEL = 3;
const int Bitmap::FILE_HEADER_SIZE = 14;
const int Bitmap::INFO_HEADER_SIZE = 40;

//  онструктор, создающий 2D массив дл€ каждого пиксел€ со значением цвета в RGB-палитре
Bitmap::Bitmap(int Height, int Width, const char* ImageName) :
	height(Height), width(Width), imageFileName(ImageName) {
	paddingAmount = ((4 - (width * 3) % 4) % 4); // ¬ычисл€ет смещение дл€ каждой 'строки' пикселей

	image = new Colour * [height];

	for (int i = 0; i < height; i++) {
		image[i] = new Colour[width];
	}
}

Bitmap::Bitmap(const char* fileName){
	std::ifstream inputFile(fileName, std::ios_base::binary);

	if (inputFile) {
		inputFile.read(reinterpret_cast<char*>(), FILE_HEADER_SIZE);
	}
}

// ƒеструктор, который будет уничтожать созданный объект и освобождать зан€тую им пам€ть
Bitmap::~Bitmap() {
	for (int i = 0; i < height; i++) {
		delete[] image[i];
	}
	delete[] image;
}

// ѕолучаем значение каждого индивидуального пиксел€
Colour Bitmap::GetColour(int x, int y) const {
	return image[y][x];
}

// ”станавливаем значение каждого пиксел€
void Bitmap::SetColour(const Colour& colour, int x, int y) {
	image[y][x].red = colour.red;
	image[y][x].green = colour.green;
	image[y][x].blue = colour.blue;
}

// √енерирует .bmp файл
void Bitmap::GenerateBitmapImage() const {
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
			unsigned char r = static_cast<unsigned char>(GetColour(x, y).red * 255.0f);		// красный канал - число с плавающей точкой
			unsigned char g = static_cast<unsigned char>(GetColour(x, y).green * 255.0f);	// поэтому умножаем на значение 255.0f,
			unsigned char b = static_cast<unsigned char>(GetColour(x, y).blue * 255.0f);	// чтобы попасть в интервал (0; 255)

			unsigned char pixelColour[] = { b, g, r }; // пиксели записываютс€ с синего канала THIS IS THE WAY

			file.write(reinterpret_cast<char*>(pixelColour), 3);
		}

		file.write(reinterpret_cast<char*>(bmpPad), paddingAmount);
	}

	file.close();

	std::cout << "File has been created!\n";
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