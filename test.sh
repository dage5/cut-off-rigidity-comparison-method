values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

for i in "${values[@]}"
do
	:
	python3 cmp_f.py $i
	echo =====================================
done
