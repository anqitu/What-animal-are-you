def ensure_directory(path):
	import os
	if not os.path.exists(path):
		os.mkdir(path)

def get_subfolder_names(folder_path):
	import os
	return next(os.walk(folder_path))[1]

def get_subfolder_paths(folder_path):
	import os
	return [os.path.join(folder_path, sub) for sub in next(os.walk(folder_path))[1]]

def get_sub_fnames(folder_path):
	import os
	return [f for f in next(os.walk(folder_path))[2] if f!= '.DS_Store']

def get_sub_fpaths(folder_path):
	import os
	return [os.path.join(folder_path,f) for f in next(os.walk(folder_path))[2] if f!= '.DS_Store']

def get_fname_from_path(path):
	return path.split('/')[-1].split('.')[0]

def get_now():
    from datetime import datetime
    print()
    print('='*50)
    print(str(datetime.now()))

def read_image_from_url(url, target_size = None):
    try:
        import requests, io
        from PIL import Image
        r = requests.get(url, timeout=15)
        img = Image.open(io.BytesIO(r.content))
        if target_size != None: img = img.resize(target_size, Image.ANTIALIAS)
        return img
    except:
        print("{:<10} Cannot find image from {}".format('[ERROR]', url))
        exit(1)

def read_image_from_path(image_path, target_size = None):
    try:
        from PIL import Image
        img = Image.open(image_path)
        if target_size != None: img = img.resize(target_size, Image.ANTIALIAS)
        return img
    except:
        print("{:<10} Cannot find image from {}".format('[ERROR]', image_path))
        exit(1)

def display_one_image(img, text = ''):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.title(text, size = 20)
    plt.show()


def display_two_images(images, text = ''):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	fig.subplots_adjust(wspace=0, hspace=0)
	for i in range(2):
		ax = fig.add_subplot(1, 2, i + 1, xticks=[], yticks=[])
		ax.imshow(crop_image_from_center(images[i]), cmap=plt.cm.binary, interpolation='nearest')
		plt.axis('off')
		plt.tight_layout()
	plt.suptitle(text, size = 20)
	# ax.text(7, 7, str(y_images[i]))
	plt.show()

# Pad the image to a square
def pad_image_to_square(img):
    from PIL import Image
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2
    img_pad = img.crop((
        -horizontal_padding,
        -vertical_padding,
        img.size[0] + horizontal_padding,
        img.size[1] + vertical_padding))
    return img_pad

def crop_image_from_center(img):
    from PIL import Image
    side = min(img.size[0], img.size[1])
    half_the_side = side / 2
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    img_crop = img.crop((
        half_the_width - half_the_side,
        half_the_height - half_the_side,
        half_the_width + half_the_side,
        half_the_height + half_the_side))
    return img_crop

def plot_prob(probs, ordered_labels):
    import matplotlib.pyplot as plt
    plt.figure(num = 1)
    plt.barh((range(0, len(ordered_labels))), probs, alpha=0.5)
    plt.yticks((range(0, len(ordered_labels))), ordered_labels)
    plt.xlabel('Similarity with All Animals')
    plt.xlim(0,1.01)
    plt.tight_layout()
    plt.show()


def plot_prob_radar(probs, ordered_labels, title = 'Probability'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    angles=np.linspace(0, 2*np.pi, len(ordered_labels), endpoint=False)
    probs=np.concatenate((probs,[probs[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    sns.set()

    ax = plt.subplot(1,1,1, polar=True)   # Set polar axis
    ax.plot(angles, probs, 'o-', linewidth=2)  # Draw the plot (or the frame on the radar chart)
    ax.fill(angles, probs, alpha=0.25)  #Fulfill the area
    ax.set_thetagrids(angles * 180/np.pi, ordered_labels)  # Set the label for each axis
    ax.set_title(title, size = 20)
    ax.set_rlim(0,1)
    ax.grid(True)
    plt.show()
