function [striped_image,stripe_locations] = make_stripes(original_image, d, r)

[num_rows num_cols num_bands] = size(original_image);

num_stripes = round(d * num_cols * num_bands);

stripe_locations = randperm(num_cols * num_bands, num_stripes);

stripes = repmat(r * (2 * rand(1, num_stripes) - 1), [num_rows 1]);

striped_image = original_image;

striped_image(:, stripe_locations) = ...
    original_image(:, stripe_locations) + stripes;

striped_image(striped_image > 1) = 1-0.001;
striped_image(striped_image < 0) = 1e-3;
