#ifndef BITMAP_H
#define BITMAP_H


// Structure that contains colour of each pixels
struct Colour {
public:
	float		red;
	float		green;
	float		blue;

	Colour();
	Colour(float r, float g, float b);
	~Colour();
};


// Class that creates and changes a .bmp file
class Bitmap {
public:
	Bitmap() = delete;
	Bitmap(int Height, int Width, const char* ImageName = "test.bmp");
	Bitmap(const char* inputFileName, const char* outputFileName = "test.bmp");
	~Bitmap();

	Colour					GetColour(int x, int y) const;
	void					SetColour(const Colour& colour, int x, int y);
	int						GetImageHeight() const;
	int						GetImageWidth() const;
	int						GetImageSize() const;
	float*					GenerateLinearizedPixelMap() const;
	float**					Generate2DPixelMap() const;
	void					InitializeWithLinearizedPixelMap(float* pixelMap, const int pixelMapHeight, const int pixelMapWidth);
	void					InitializeWith2DPixelMap(float** pixelMap, const int pixelMapHeight, const int pixelMapWidth);
	void					GenerateBitmapImage(const char* fileName);
	void					DisplayImageInfo() const;

private:
	static const int		BYTES_PER_PIXEL;
	static const int		FILE_HEADER_SIZE;
	static const int		INFO_HEADER_SIZE;

	int						height;
	int						width;
	int						size;
	int						paddingAmount;

	Colour**				image;
	const char*				imageFileName;

	unsigned char*			CreateBitmapFileHeader() const;
	unsigned char*			CreateBitmapInfoHeader() const;
	void					ClearImageData();
	void					FreePixelMapMemory(float** pixelMap, int pixelMapHeight, int pixelMapWidth);
};


#endif // !BITMAP_H
