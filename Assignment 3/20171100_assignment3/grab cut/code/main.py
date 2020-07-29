import matplotlib.pyplot as plt
import maxflow
import numpy as np
from os import listdir
import pdb
import random
from scipy.misc import imread, imsave
from scipy.spatial.distance import cdist

import time


Kmeans_thresh = 1.0
max_iter = 7 
thresh = 0.001
KMEANS_COMPONENTS = 4
fact = 50

SHIFT_DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP_LEFT', 'UP_RIGHT']

STRUCTURES = {
	'UP': np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
	'DOWN': np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
	'LEFT': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
	'RIGHT': np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
	'UP_LEFT': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
	'UP_RIGHT': np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
}

def auto_grabcut():
	box_list, data_list = sorted(['../bboxes/' + f for f in listdir('../bboxes')]), sorted(['../images/' + f for f in listdir('../images/')])
	
	mean_accuracy, mean_similarity = 0.0, 0.0


	for box_path, img_path in zip(box_list, data_list):
		with open(box_path) as box_file:
			bounds = box_file.readline().split()
			box = {
				'x_min': int(bounds[0]),
				'y_min': int(bounds[1]),
				'x_max': int(bounds[2]),
				'y_max': int(bounds[3])
			}

		print('SEGMENTING IMAGE {}'.format(img_path))
		img = imread(img_path)

		seg = grabcut(img, box)
		seg = 1- seg
		out = np.zeros(img.shape)
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if seg[i,j] == 0:
					out[i,j,:] = img[i,j,:]

		imsave('../report/Outputs/' + img_path.split('/')[2], out)
		

def box_grabcut(img_path,box):

	print('SEGMENTING IMAGE {}'.format(img_path))
	img = imread(img_path)
	seg = grabcut(img, box)
	seg = 1-seg
	out = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if seg[i,j] == 0:
				out[i,j,:] = img[i,j,:]
	imsave('./output/' + img_path.split('/')[2], out)
	
def get_accuracy(seg_map, ground_truth):
	correct = (seg_map + ground_truth - 1) ** 2
	return np.mean(correct)

def grabcut(img, box):
	
	seg_map = np.zeros((img.shape[0], img.shape[1]))
	seg_map[int(box['y_min']) - 1:int(box['y_max']) + 1, int(box['x_min']) - 1:int(box['x_max']) + 1] = 1
	
	bg, fg = img[seg_map == 0], img[seg_map == 1]
	fg_ass, bg_ass = quick_k_means(fg, bg)

	change = 1
	i = 0
	oldshape = fg.shape[0]
	while (change > thresh) and (i < max_iter):
		fg_gmm, bg_gmm = fit_gmm(fg, bg, fg_ass, bg_ass)
		seg_map, fg_ass, bg_ass = estimate_segmentation(img, fg_gmm, bg_gmm, seg_map, box)

		bg,fg = img[seg_map == 0], img[seg_map == 1]
		change = abs(oldshape-fg.shape[0])/float(oldshape)
		oldshape = fg.shape[0]
		
		i += 1
	return seg_map

def adjust_outside_box(fg_unary,val, box):
	fg_unary[:int(box['y_min']), :] =  fg_unary[:, :int(box['x_min'])] = fg_unary[int(box['y_max']):, :] = fg_unary[:, int(box['x_max']):] = val
	return fg_unary

def estimate_segmentation(img, fg_gmm, bg_gmm, seg_map, box):
	fg_unary, fg_ass = get_unary(img, fg_gmm)
	bg_unary, bg_ass = get_unary(img, bg_gmm)
	bg_unary ,fg_unary = adjust_outside_box(bg_unary,0,box), adjust_outside_box(fg_unary,10000,box)

	# Calculate pairwise values
	pair_pot = get_pairwise(img)

	
	pot_graph, nodes = create_graph(fg_unary, bg_unary, pair_pot)
	pot_graph.maxflow()


	box_seg = segment(pot_graph, nodes)

	seg_map = np.zeros((img.shape[0], img.shape[1]), dtype='int32')
	seg_map[int(box['y_min'])-1:int(box['y_max'])+1, int(box['x_min'])-1:int(box['x_max'])+1] = box_seg[int(box['y_min'])-1:int(box['y_max'])+1, int(box['x_min'])-1:int(box['x_max'])+1]
	
	return box_seg, fg_ass[box_seg == 1], bg_ass[box_seg == 0]

def create_graph(fg_unary, bg_unary, pair_pot):
	graph = maxflow.Graph[float]()
	nodes = graph.add_grid_nodes(fg_unary.shape)


	graph.add_grid_tedges(nodes, fg_unary, bg_unary)


	for i, direction in enumerate(SHIFT_DIRECTIONS):
		if i in [0, 2, 4, 5]:
			graph.add_grid_edges(nodes, weights=pair_pot[i], structure=STRUCTURES[direction], symmetric=False)

	return graph, nodes

def segment(graph, nodes):
	segments = graph.get_grid_segments(nodes)
	return (np.int_(np.logical_not(segments)) - 1) * -1

def get_pairwise(img):
	H, W, C = img.shape

	shifted_imgs = shift(img)
	pairwise_dist = np.zeros((6, H, W))

	for i in range(6):
		pairwise_dist[i] = np.sqrt(np.sum((img - shifted_imgs[i]) ** 2, axis=2))

	beta = 1.0 / (2 * np.mean(pairwise_dist[[0,2,4,5],:,:]))

	pairwise_dist = np.exp(-1 * beta * pairwise_dist)
	pairwise_dist *= fact

	return pairwise_dist

def shift(img):
	H, W, C = img.shape

	up = down = left = right = up_left = up_right = np.array(img)
	up[:H-1, : ,:], down[1:, :, :], left[:, :W-1, :], right[:, 1:, :] = img[1:, :, :], img[:H-1, :, :], img[:, 1:, :], img[:, :W-1, :]
	up_left[:H-1, :W-1, :] = img[1:, 1:, :]
	up_right[:H-1, 1:, :] = img[1:, :W-1, :]

	shifted_imgs = np.zeros((6, H, W, C), dtype='uint32')
	shifted_imgs[0],shifted_imgs[1],shifted_imgs[2],shifted_imgs[3],shifted_imgs[4],shifted_imgs[5]  = up,down,left,right,up_left,up_right

	return shifted_imgs

def get_unary(img, gmm):
	K = len(gmm)
	H, W, C = img.shape

	potentials = np.zeros((K, H, W))
	log_pdfs = np.zeros((K, H, W), dtype='float64')

	for k, gaussian in enumerate(gmm):
		cov = gaussian['cov']
		mu_img = img - np.reshape(gaussian['mean'], (1, 1, 3))

		log_pdfs[k] +=  -0.5*np.log(np.linalg.det(cov))

		piece1 = -np.log(gaussian['size']) + 0.5 * np.log(np.linalg.det(cov))
		temp = np.einsum('ijk,il', np.transpose(mu_img), np.linalg.inv(cov))
		
		piece2 = np.zeros((H, W))
		
		for i in range(H):
			for j in range(W):
				piece2[i, j] = np.dot(temp[j, i], mu_img[i, j])

		piece2 *= 0.5
		potentials[k] = piece1 + piece2
		log_pdfs[k] += -1.0 * piece2
	
	assignments = np.argmax(np.array(log_pdfs), axis=0)
	unary = np.zeros((H,W))
	for i in range(H):
		for j in range(W):
			unary[i,j] = potentials[assignments[i,j],i,j]

	return unary, assignments

def quick_k_means(foreground, background, k=KMEANS_COMPONENTS):
	fg_mu, bg_mu = foreground[np.random.choice(foreground.shape[0], k), :], background[np.random.choice(background.shape[0], k), :] 
	avg_centroid_change = float('Inf')

	while avg_centroid_change > Kmeans_thresh:
		fg_dist = cdist(foreground, fg_mu, metric='euclidean')
		fg_ass, new_fg_mu = np.argmin(fg_dist, axis=1), np.zeros_like(fg_mu)
		
		bg_dist = cdist(background, bg_mu, metric='euclidean')
		bg_ass, new_bg_mu = np.argmin(bg_dist, axis=1), np.zeros_like(bg_mu)

		for i in range(k):
			new_fg_mu[i],new_bg_mu[i] = np.mean(foreground[fg_ass == i], axis=0),np.mean(background[bg_ass == i], axis=0)


		avg_centroid_change = np.mean(np.sqrt(np.sum(np.square(fg_mu - new_fg_mu), axis=1))) 

		fg_mu, bg_mu = new_fg_mu, new_bg_mu

	return fg_ass, bg_ass

def fit_gmm(fg, bg, fg_ass, bg_ass, k=KMEANS_COMPONENTS):
	fg_gmms, bg_gmms = [], []

	for i in range(max(np.max(fg_ass), np.max(bg_ass)) + 1):
		fg_gmm = create_gmm(fg,fg_ass,i)
		if fg_gmm['size'] > 0.001:
			fg_gmms.append(fg_gmm)
		
		bg_gmm = create_gmm(bg,bg_ass,i)
		if bg_gmm['size'] > 0.001:
			bg_gmms.append(bg_gmm)
				
		
	if len(fg_gmms) < k:
		split(fg_gmms, fg, fg_ass, k)
	
	return fg_gmms, bg_gmms

def create_gmm(fg,fg_ass,i):
	fg_cluster = fg[fg_ass == i]
	fg_gmm = {}
	fg_gmm['mean'] = np.mean(fg_cluster, axis=0)
	fg_gmm['cov'] = np.cov(fg_cluster, rowvar=0) + np.identity(3)*1e-8
	fg_gmm['size'] = fg_cluster.shape[0] * 1.0 / fg.shape[0]
	return fg_gmm




def split(gmm_list, pixels, assignment, k):
	sizes = np.array([f['size'] for f in gmm_list])
	gmm_list.pop(np.argmax(sizes))
	orig_size = np.max(sizes)
	
	largest = np.argmax(np.bincount(assignment))
	members = pixels[assignment == largest]

	num_new_comps = k - len(gmm_list)
	ass1, ass2 = quick_k_means(members, members, k=num_new_comps)

	for i in range(num_new_comps):
		new_members = members[ass1 == i]

		new_gmm = {
			'mean': np.mean(new_members, axis=0),
			'cov': np.cov(new_members, rowvar=0) + np.identity(3)*1e-8,
			'size': orig_size * new_members.shape[0] / members.shape[0]
		}

		gmm_list.append(new_gmm)

	return gmm_list

def select_bounding_box(img):
	img = imread(img)
	plt.imshow(img)

	click = plt.ginput(2)
	plt.close()
	
	box = {
		'x_min': int(round(min(click[0][0], click[1][0]))),
		'y_min': int(round(min(click[0][1], click[1][1]))),
		'x_max': int(round(max(click[0][0], click[1][0]))),
		'y_max': int(round(max(click[0][1], click[1][1]))),
	}
	
	width = float(img.shape[1])
	x_range_min = int(box['x_min'] / width)
	x_range_max = int(box['x_max'] / width)

	plt.axhspan(box['y_min'], box['y_max'], x_range_min, x_range_max, color='r', fill=False)
	plt.imshow(img)
	plt.show()

	return box

if __name__ == '__main__':
	import sys
	try:
	     # Just in case
	    start = sys.version.index('|') # Do we have a modified sys.version?
	    end = sys.version.index('|', start + 1)
	    version_bak = sys.version # Backup modified sys.version
	    sys.version = sys.version.replace(sys.version[start:end+1], '') # Make it legible for platform module
	    import platform
	    platform.python_implementation() # Ignore result, we just need cache populated
	    platform._sys_version_cache[version_bak] = platform._sys_version_cache[sys.version] # Duplicate cache
	    sys.version = version_bak # Restore modified version string
	except ValueError: # Catch .index() method not finding a pipe
		pass

	if len(sys.argv) > 1:
		img_path = sys.argv[1]

		box = select_bounding_box(img_path)
		box_grabcut(img_path,box)
	else:
		auto_grabcut()
