function output = PatchAugmentation(input, ind)

if (ind == 1)
    output = input;
end

if (ind == 2)
    output = rot90(input, 1);
end

if (ind == 3)
    output = rot90(input, 2);
end

if (ind == 4)
    output = fliplr(input);
end

if (ind == 5)
    output = flipud(input);
end

if (ind == 6)
    output = rot90(fliplr(input), 1);
end

if (ind == 7)
    output = rot90(fliplr(input), 1);
end

if (ind == 8)
    output = rot90(flipud(input), 2);
end

if (ind == 9)
    output = rot90(flipud(input), 2);
end
