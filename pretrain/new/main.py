from new import make_dataloader
train_loader, train_loader_normal, val_loader, corrupted_val_loader, corrupted_query_loader, corrupted_gallery_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

