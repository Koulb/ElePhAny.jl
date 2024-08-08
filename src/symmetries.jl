function fold_component(x, eps=5e-3)
    """
    This routine folds number with given accuracy, so it would be inside the section from 0 to 1 .

        Returns:
            :x: folded number

    """
    if x >= 1 - eps
        while x >= 1 - eps
            x = x - 1
        end
    elseif x < 0 - eps
        while x < 0 - eps
            x = x + 1
        end
    end
    return x
end

function rotate_grid(N1, N2, N3, rot, tras)
    """
    This routine change the grid according to given rotation and translation.

        Returns:
            :mapp (list): list of indexes of transformed grid

    """
    mapp = []
    for k in 0:N3-1
        for j in 0:N2-1
            for i in 0:N1-1
                u = [i / N1, j / N2, k / N3]
                ru = rot * u .+ tras
                ru[1] = fold_component(ru[1])
                ru[2] = fold_component(ru[2])
                ru[3] = fold_component(ru[3])

                i1 = round(Int, ru[1] * N1)
                i2 = round(Int, ru[2] * N2)
                i3 = round(Int, ru[3] * N3)

                eps = 1e-5
                if i1 >= N1 - eps || i2 >= N2 - eps || i3 >= N3 - eps
                    #println(i1, i2, i3, N1, N2, N3)
                    # error("Error in folding")
                end

                ind = i1 + (i2) * N1 + (i3) * N1 * N2 
                push!(mapp, ind)
            end
        end
    end
    return mapp
end

function rotate_deriv(N1, N2, N3, mapp, ff)
    """
    This routine rotate the derivative according to the given grid.

        Returns:
            :ff_rot (np.array): array containing values of the derivative on a new frid

    """
    ff_rot = zeros(ComplexF64, N1, N2, N3)
    ind = 1
    for k in 0:N3-1
        for j in 0:N2-1
            for i in 0:N1-1
                ind1 = mapp[ind]
                i3 = div(ind1, N2 * N1)
                ind1 = ind1 % (N1 * N2)
                i2 = div(ind1, N1)
                i1 = ind1 % N1
               # if (i1 + (i2) * N1 + (i3) * N1 * N2) != mapp[ind]
                    #println("different")
                    #println(i1, i2, i3, ind, mapp[ind])
                    # error()
                #end
                ind += 1

                ff_rot[i1+1, i2+1, i3+1] = ff[i+1, j+1, k+1]
            end
        end
    end
    return ff_rot
end
