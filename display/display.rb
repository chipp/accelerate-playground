require 'RMagick'
include Magick

images_ios = File.new '../train-images.idx3-ubyte', 'rb'
msb = images_ios.read(4).unpack('H8').first.hex.to_i
images_count = images_ios.read(4).unpack('H8').first.hex.to_i
rows = images_ios.read(4).unpack('H8').first.hex.to_i
cols = images_ios.read(4).unpack('H8').first.hex.to_i

labels_ios = File.new '../train-labels.idx1-ubyte', 'rb'
msb = labels_ios.read(4).unpack('H8').first.hex.to_i
labels_count = labels_ios.read(4).unpack('H8').first.hex.to_i

image_list = Magick::ImageList.new

i = 0

until images_ios.eof?
  image = image_list.new_image(cols, rows)

  rows.times do |row|
    cols.times do |column|
      value = images_ios.readbyte
      value *= 255
      pixel = Pixel.new value, value, value
      image.pixel_color column, row, pixel
    end
  end
end
