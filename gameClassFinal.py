import pygame
from random import randint
from DQNFinal import DQNAgent
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set options to activate or deactivate the game view, and its speed
display_option = False
speed = 0
pygame.font.init()


class Game:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width+40, game_height+60))
        self.bg = pygame.image.load("img/background2.png")
        self.crash = False
        self.player = Player(self)
        self.block = block()#.position[0]#block()
        self.blocks = block().position
        self.score = 0


class Player(object):

    def __init__(self, game):
        x = 20                        #start at beginning
        y = 0.5 * game.game_height    #start mid height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.block = 1
        self.passed = False
        self.forward = 0
        self.image = pygame.image.load('img/player.png')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, block,agent):

        if self.passed:
            self.passed = False
            self.block = self.block + 1
        if np.array_equal(move ,[1, 0, 0]):
            self.y_change = 0                  #move forward
        elif np.array_equal(move,[0, 1, 0]):   #move up
            self.y_change = 20
        elif np.array_equal(move,[0, 0, 1]) :  #move down
            self.y_change = -20
        if self.y_change == 0:
           self.x = x + self.x_change          #move forward
        else:
           self.x = x 
        if self.x == 20:                       #go forward at start
           self.y = y
           self.x = x + self.x_change
        elif self.y + self.y_change > game.game_height-10: #dont go past boundries
           self.y = y 
        elif self.y + self.y_change < 20:                  #dont go past boundries
           self.y = y 
        else:
           self.y = y + self.y_change
        if self.x > game.game_width-40:   #if reached end, start over
            self.x = 20

        if [self.x,self.y] in block.position:       #crash into block 
            game.crash = True
        if self.y_change == 0:
            self.forward = 1
        else:
            self.forward = 0
        reach_end(self, block, game)

        self.update_position(self.x, self.y)

    def display_player(self, x, y, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            x_temp, y_temp = self.position[len(self.position)-1]
            game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class block(object):

    def __init__(self):
        self.x_block = 240   #initialize starting block ~mid screen
        self.y_block = 40
        self.image = pygame.image.load('img/barrier.png')
        self.position = []
        self.position.append([self.x_block, self.y_block]) #add blocks to list

    def block_coord(self, game, player):  #generate random new block position
        x_rand = randint(60, game.game_width - 40) 
        self.x_block = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height-10)
        self.y_block = y_rand - y_rand % 20
        if [self.x_block, self.y_block] not in player.position and [self.x_block, self.y_block] not in self.position:
            return self.x_block, self.y_block  #dont repeat blocks
        else:
            self.block_coord(game,player)

    def display_block(self, x, y, game):
        
        if game.crash == False:
            for i in range(len(self.position)):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


def reach_end(player, block, game):
    if player.x == 20:    #reached end ; set back to start = 1 point
        block.block_coord(game, player)
        player.passed = True
        game.score = game.score + 1
        block.position.append([block.x_block, block.y_block])  #new block added to game


def get_record(score, record):
        if score >= record:
            return score
        else:
            return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, block, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], game)
    block.display_block(block.x_block, block.y_block, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, block, blocks, agent):
    state_init1 = agent.get_state(game, player, blocks)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, block, agent)
    state_init2 = agent.get_state(game, player, blocks)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory)


def plot_seaborn(array_counter, array_score): #get performance measures
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'blue'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()

def run():
    pygame.init()
    agent = DQNAgent() #use agent from DQN.py
    counter_games = 0
    score_plot = []
    counter_plot =[]
    record = 0
    while counter_games < 500:
        # Initialize classes
        game = Game(440, 100) 
        player1 = game.player
        block1 = game.block
        blocks1 = game.blocks

        # Perform first move
        initialize_game(player1, game, block1, blocks1, agent)
        if display_option:
            display(player1, block1, game, record)

        while not game.crash:
            #agent.epsilon is set to give randomness to actions
            agent.epsilon = 80 - counter_games
            
            #get old state
            state_old = agent.get_state(game, player1, blocks1)
            
            #perform random actions based on agent.epsilon, or choose the action
            if randint(0, 200) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1,5)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)
                
            #perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, block1, agent)
            state_new = agent.get_state(game, player1, blocks1)
            
            #set reward for the new state
            reward = agent.set_reward(player1, game.crash)
            
            #train short memory base on the new action and state
            agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
            
            # store the new data into a long term memory
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            record = get_record(game.score, record)
            if display_option:
                display(player1, block1, game, record)
                pygame.time.wait(speed)
        
        agent.replay_new(agent.memory)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.score)
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    agent.model.save_weights('weights4.hdf5')
    plot_seaborn(counter_plot, score_plot)


run()
