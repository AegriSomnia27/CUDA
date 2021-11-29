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
	Bitmap(const char* inputFileName, const char* outputFileName);
	~Bitmap();

	// methods on CPU
	Colour					GetColour(int x, int y) const;
	void					SetColour(const Colour& colour, int x, int y);
	int						GetImageHeight() const;
	int						GetImageWidth() const;
	void					GenerateBitmapImage(const char* fileName);
	void					DisplayImageInfo() const;

private:
	static const int		BYTES_PER_PIXEL;
	static const int		FILE_HEADER_SIZE;
	static const int		INFO_HEADER_SIZE;

	int						height;
	int						width;
	int						paddingAmount;

	Colour**				image;
	const char*				imageFileName;

	unsigned char*			CreateBitmapFileHeader() const;
	unsigned char*			CreateBitmapInfoHeader() const;
};


#endif // !BITMAP_H
