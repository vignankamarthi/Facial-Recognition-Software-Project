# Test Face Images

## Required Images

To run the test suite, you need to provide several real face images in this directory:

1. **real_face.jpg**: Basic face image for general tests
2. **reference_face.jpg**: Face image used as a known reference face
3. **test_face.jpg**: Face image used for testing matching against reference faces
4. **anon_face.jpg**: Face image for testing anonymization (optional, will use test_face.jpg if not provided)

## Image Requirements

- Clear frontal faces for best results
- Good lighting
- Minimal background complexity
- Ideally 500x500 pixels or larger

## Privacy Considerations

- Consider using publicly available face images or faces with appropriate permissions
- Alternatively, you can use your own photos since these are only for local testing

## Test Skipping

If these images are not found, relevant tests will be automatically skipped with messages indicating which files are missing.
