
service FaceEmbedding {
	list< list<double> > get_emb_raw(1:list<binary> raw_imgs),
	list< list<double> > get_emb_numpy(1:list<binary> numpy_imgs)
}