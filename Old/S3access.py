from boto.s3.connection import S3Connection

conn = S3Connection('mvukicevic@censia.com','Nalizesse83')
bucket = conn.get_bucket('bucket')
for key in bucket.list():
    try:
        res = key.get_contents_to_filename(key.name)
    except:
        logging.info(key.name+":"+"FAILED")
