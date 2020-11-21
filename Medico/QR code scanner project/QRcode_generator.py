#import os
import qrcode
#from PIL import Image
qr = qrcode.QRCode(version = 1, box_size = 10, border = 5)
data = "Hello I am A QR CODE"
qr.add_data(data,)
qr.make(fit = True)
img = qr.make_image(fill = "black", back_color = "white")
img = img.convert('RGB')
img.save("Generated QR Code.pdf")